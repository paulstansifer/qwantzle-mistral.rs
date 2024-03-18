use std::{iter::zip, ops::Mul};

use candle_core::{Module, Result, Shape, Tensor};
use candle_nn::{init, Dropout, Linear, VarBuilder};
use either::Either;

use crate::{
    apply_scalings_to_x, frozenlinear::FrozenLinear, get_maybe_topk_scalings, LinearLayerLike,
    LoraConfig, LoraLinearConfig,
};

#[derive(Debug)]
pub struct LoraLinear {
    old: FrozenLinear,
    a_adapters: Either<Vec<Linear>, (Tensor, Vec<Linear>)>,
    b_adapters: Either<Vec<Linear>, (Linear, Vec<Linear>)>,
    scale_adapters: Either<Vec<f64>, (Tensor, Vec<f64>)>,
    dropout_adapters: Vec<Option<Dropout>>,
    layer_n: usize,
}

impl LoraLinear {
    pub fn new(
        old: &dyn LinearLayerLike,
        linear_config: &LoraLinearConfig,
        config: &[(String, LoraConfig)],
        vb: &VarBuilder,
        layer_n: usize,
    ) -> Result<Self> {
        let mut a_adapters = Vec::with_capacity(config.len());
        let mut b_adapters = Vec::with_capacity(config.len());
        let mut scale_adapters = Vec::with_capacity(config.len());
        let mut dropout_adapters = Vec::with_capacity(config.len());
        let a_vb = vb.pp("lora_A".to_string());
        let b_vb = vb.pp("lora_B".to_string());
        let mut state = None;
        let mut all_same = true;
        for (name, cfg) in config.iter() {
            let a_pp = a_vb.pp(name);
            assert!(a_pp.contains_tensor("weight"));
            let a = a_pp.get_with_hints(
                (cfg.rank, linear_config.in_features),
                "weight",
                init::DEFAULT_KAIMING_NORMAL,
            )?;
            let b_pp = b_vb.pp(name);
            assert!(b_pp.contains_tensor("weight"));
            let b =
                b_pp.get_with_hints((linear_config.out_features, cfg.rank), "weight", init::ZERO)?;
            a_adapters.push(Linear::new(a, None));
            b_adapters.push(Linear::new(b, None));
            scale_adapters.push(if cfg.rank > 0 {
                cfg.alpha / cfg.rank as f64
            } else {
                1.0
            });
            dropout_adapters.push(cfg.dropout.map(Dropout::new));
            if state.is_some_and(|x| {
                x == (
                    cfg.rank,
                    linear_config.in_features,
                    linear_config.out_features,
                    cfg.alpha,
                    cfg.dropout,
                )
            }) || state.is_none()
            {
                state = Some((
                    cfg.rank,
                    linear_config.in_features,
                    linear_config.out_features,
                    cfg.alpha,
                    cfg.dropout,
                ));
            } else {
                all_same = false;
            }
        }

        if all_same {
            let a_adapters_stack = Tensor::cat(
                &a_adapters
                    .iter()
                    .map(|x| x.weight().unsqueeze(0))
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            let b_adapters_stack = Tensor::cat(
                &b_adapters
                    .iter()
                    .map(|x| x.weight().unsqueeze(0))
                    .collect::<Result<Vec<_>>>()?,
                0,
            )?;
            Ok(LoraLinear {
                old: FrozenLinear::new_from_linear(old)?,
                a_adapters: Either::Right((a_adapters_stack.clone(), a_adapters)),
                b_adapters: Either::Right((Linear::new(b_adapters_stack, None), b_adapters)),
                scale_adapters: Either::Right((
                    Tensor::from_vec(
                        scale_adapters.clone(),
                        scale_adapters.len(),
                        a_adapters_stack.device(),
                    )?
                    .unsqueeze(1)?
                    .unsqueeze(1)?
                    .to_dtype(a_adapters_stack.dtype())?,
                    scale_adapters,
                )),
                dropout_adapters,
                layer_n,
            })
        } else {
            Ok(LoraLinear {
                old: FrozenLinear::new_from_linear(old)?,
                a_adapters: Either::Left(a_adapters),
                b_adapters: Either::Left(b_adapters),
                scale_adapters: Either::Left(scale_adapters),
                dropout_adapters,
                layer_n,
            })
        }
    }
}

impl LinearLayerLike for LoraLinear {
    fn bias(&self) -> Option<&Tensor> {
        self.old.bias()
    }
    fn weight(&self) -> &Tensor {
        self.old.weight()
    }
    fn shape(&self) -> &Shape {
        self.old.shape()
    }
    fn lora_forward(
        &self,
        input: &Tensor,
        scalings: Tensor,
        global_scaling_weight: f64,
        is_scaling_pass: Option<f64>,
    ) -> Result<Tensor> {
        let mut result = self.old.forward(input)?;

        if is_scaling_pass.is_some_and(|x| x == 0.) {
            return Ok(result);
        }

        let scalings = get_maybe_topk_scalings(scalings, self.layer_n)?;
        if self.a_adapters.is_left() || scalings.dims3()?.1 != 1 {
            let a_adapters = if self.a_adapters.is_right() {
                self.a_adapters.as_ref().unwrap_right().1.clone()
            } else {
                self.a_adapters.as_ref().unwrap_left().clone()
            };
            let b_adapters = if self.b_adapters.is_right() {
                self.b_adapters.as_ref().unwrap_right().1.clone()
            } else {
                self.b_adapters.as_ref().unwrap_left().clone()
            };
            let scale_adapters = if self.scale_adapters.is_right() {
                self.scale_adapters.as_ref().unwrap_right().1.clone()
            } else {
                self.scale_adapters.as_ref().unwrap_left().clone()
            };
            //No fan_in_fan_out so no weight.transpose(0,1)
            for (i, (adapter_a, (adapter_b, (adapter_scale, adapter_dropout)))) in zip(
                a_adapters,
                zip(b_adapters, zip(scale_adapters, &self.dropout_adapters)),
            )
            .enumerate()
            {
                let mut input_new = input.to_dtype(adapter_a.weight().dtype())?;
                input_new = apply_scalings_to_x(input_new.clone(), &scalings, i)?;

                input_new = if let Some(ref dropout) = adapter_dropout {
                    dropout.forward(&input_new, true)?
                } else {
                    input_new.clone()
                };

                let res = adapter_b
                    .forward(&adapter_a.forward(&input_new)?)?
                    .mul(adapter_scale)?
                    .mul(global_scaling_weight)?;
                result = (result + res)?;
            }
            Ok(result)
        } else {
            let adapter_a = &self.a_adapters.as_ref().unwrap_right().0;
            let adapter_b = &self.b_adapters.as_ref().unwrap_right().0;
            let (adapter_scales, vec) = &self.scale_adapters.as_ref().unwrap_right();
            let n_adapters = vec.len();
            let dropout = &self.dropout_adapters[0];
            let scalings = scalings
                .squeeze(0)?
                .squeeze(0)?
                .unsqueeze(1)?
                .unsqueeze(1)?
                .broadcast_mul(adapter_scales)?;
            let adapter_a = adapter_a
                .broadcast_mul(&scalings)?
                .mul(global_scaling_weight)?;

            let input = if let Some(ref d) = dropout {
                d.forward(input, true)?
            } else {
                input.clone()
            };
            let (b, s, h) = input.dims3()?;
            let input = input.reshape((b * s, h))?;
            let out = adapter_a.broadcast_matmul(&input.t()?)?;
            let out = adapter_b.weight().broadcast_matmul(&out)?;
            let o_h = out.dims()[1];
            let out = out.reshape((n_adapters, b, s, o_h))?;
            let out = out.sum(0)?;
            out + result
        }
    }
}
