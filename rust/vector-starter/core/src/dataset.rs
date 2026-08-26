use std::sync::Arc;
use crate::VectorError;

use crate::{Metric, Result};

#[derive(Debug, Clone)]
pub struct Dataset {
    vectors: Arc<[Vec<f32>]>,
    dimension: usize,
}

impl Dataset {
    pub fn try_new(_vectors: Vec<Vec<f32>>) -> Result<Self> {
        //todo!("Chapter 1: validate a non-empty, rectangular, finite dataset")
        _vectors.get(0).map(|v| v.len()).ok_or_else(|| VectorError::InvalidDimension("Dataset must have at least one vector".to_string()))?;

        let dimension = _vectors[0].len();
        for (i, vector) in _vectors.iter().enumerate() {
            if dimension != vector.len() {
                return Err(format!("Vector at index {} has inconsistent dimension, which is invalid for cosine distance.", i).into());
            }
        }

        Ok(Self {
            vectors: _vectors.into(),
            dimension: dimension,
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[row]
    }

    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    pub(crate) fn validate_for_metric(&self, _metric: Metric) -> Result<()> {
        //todo!("Chapter 1: reject zero-norm dataset rows for cosine distance")
        match _metric {
            Metric::Cosine => {
                for (i, vector) in self.vectors.iter().enumerate() {
                    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm == 0.0 {
                        return Err(format!("Vector at index {} has zero norm, which is invalid for cosine distance.", i).into());
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn validate_query(&self, _query: &[f32], _metric: Metric) -> Result<()> {
        //todo!("Chapter 1: validate query dimension, finiteness, and cosine norm")
        if _query.len() != self.dimension {
            return Err(format!("Query vector dimension {} does not match dataset dimension {}.", _query.len(), self.dimension).into());
        }

        match _metric {
            Metric::Cosine => {
                let is_valid = _query.iter().all(|&x| x.is_finite());
                if !is_valid {
                    return Err("Query vector contains non-finite values.".into());
                }
                self.validate_for_metric(_metric)?;
            }
            _ => {}
        }
    }
}
