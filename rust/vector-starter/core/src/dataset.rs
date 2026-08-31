use std::sync::Arc;
use crate::VectorError;

use crate::{Metric, Result};

#[derive(Debug, Clone)]
pub struct Dataset {
    vectors: Arc<[Vec<f32>]>,
    dimension: usize,
}

impl Dataset {
    pub fn try_new(vectors: Vec<Vec<f32>>) -> Result<Self> {
        //todo!("Chapter 1: validate a non-empty, rectangular, finite dataset")
        if vectors.is_empty() {
            return Err(VectorError::EmptyDataset);
        }
        vectors.get(0).map(|v| v.len()).ok_or_else(|| VectorError::EmptyVector)?;

        let dimension = vectors[0].len();

        for (_i, vector) in vectors.iter().enumerate() {
            if dimension != vector.len() {
                return Err(VectorError::DimensionMismatch { expected: dimension, actual: vector.len() });
            }
        }

        Ok(Self {
            vectors: vectors.into(),
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

    pub(crate) fn validate_for_metric(&self, metric: Metric) -> Result<()> {
        //todo!("Chapter 1: reject zero-norm dataset rows for cosine distance")
        match metric {
            Metric::Cosine => {
                for (i, vector) in self.vectors.iter().enumerate() {
                    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm == 0.0 {
                        return Err(VectorError::ZeroNorm { vector: i });
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn validate_query(&self, query: &[f32], metric: Metric) -> Result<()> {
        //todo!("Chapter 1: validate query dimension, finiteness, and cosine norm")
        if query.len() != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: query.len(),
            });
        }
        if let Some(dimension) = query.iter().position(|value| !value.is_finite()) {
            return Err(VectorError::NonFiniteValue {
                vector: self.len(),
                dimension,
            });
        }
        if metric == Metric::Cosine && Metric::squared_norm(query) == 0.0 {
            return Err(VectorError::ZeroNorm { vector: self.len() });
        }
        Ok(())
    }
}
