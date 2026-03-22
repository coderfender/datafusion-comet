// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Aggregate expression registry for dispatching aggregate expression creation

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion::common::internal_err;
use datafusion::physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_comet_proto::spark_expression::{agg_expr::ExprStruct as AggExprStruct, AggExpr};

use crate::execution::operators::ExecutionError;

/// Trait for building aggregate physical expressions from Spark protobuf aggregate expressions
pub trait AggregateExpressionBuilder: Send + Sync {
    /// Build a DataFusion aggregate expression from a Spark protobuf aggregate expression
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &super::PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError>;
}

/// Enum to identify different aggregate expression types for registry dispatch
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateExpressionType {
    Count,
    Min,
    Max,
    Sum,
    Avg,
    First,
    Last,
    BitAndAgg,
    BitOrAgg,
    BitXorAgg,
    Covariance,
    Variance,
    Stddev,
    Correlation,
    BloomFilterAgg,
}

/// Registry for aggregate expression builders
pub struct AggregateExpressionRegistry {
    builders: HashMap<AggregateExpressionType, Box<dyn AggregateExpressionBuilder>>,
}

impl AggregateExpressionRegistry {
    /// Create a new aggregate expression registry with all builders registered
    fn new() -> Self {
        let mut registry = Self {
            builders: HashMap::new(),
        };

        registry.register_all_aggregate_expressions();
        registry
    }

    /// Get the global shared registry instance
    pub fn global() -> &'static AggregateExpressionRegistry {
        static REGISTRY: std::sync::OnceLock<AggregateExpressionRegistry> =
            std::sync::OnceLock::new();
        REGISTRY.get_or_init(AggregateExpressionRegistry::new)
    }

    /// Check if the registry can handle a given aggregate expression type
    pub fn can_handle(&self, spark_expr: &AggExpr) -> bool {
        if let Ok(expr_type) = Self::get_aggregate_expression_type(spark_expr) {
            self.builders.contains_key(&expr_type)
        } else {
            false
        }
    }

    /// Create an aggregate expression from a Spark protobuf aggregate expression.
    /// Handles query context registration before dispatching to the builder.
    pub fn create_agg_expr(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &super::PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        // Register QueryContext if present
        register_query_context(spark_expr, planner.query_context_registry());

        let expr_type = Self::get_aggregate_expression_type(spark_expr)?;

        if let Some(builder) = self.builders.get(&expr_type) {
            builder.build(spark_expr, schema, planner)
        } else {
            Err(ExecutionError::GeneralError(format!(
                "No builder registered for aggregate expression type: {:?}",
                expr_type
            )))
        }
    }

    /// Register all aggregate expression builders
    fn register_all_aggregate_expressions(&mut self) {
        use crate::execution::expressions::aggregates::*;

        self.builders
            .insert(AggregateExpressionType::Count, Box::new(CountBuilder));
        self.builders
            .insert(AggregateExpressionType::Min, Box::new(MinBuilder));
        self.builders
            .insert(AggregateExpressionType::Max, Box::new(MaxBuilder));
        self.builders
            .insert(AggregateExpressionType::Sum, Box::new(SumBuilder));
        self.builders
            .insert(AggregateExpressionType::Avg, Box::new(AvgBuilder));
        self.builders
            .insert(AggregateExpressionType::First, Box::new(FirstBuilder));
        self.builders
            .insert(AggregateExpressionType::Last, Box::new(LastBuilder));
        self.builders
            .insert(AggregateExpressionType::BitAndAgg, Box::new(BitAndBuilder));
        self.builders
            .insert(AggregateExpressionType::BitOrAgg, Box::new(BitOrBuilder));
        self.builders
            .insert(AggregateExpressionType::BitXorAgg, Box::new(BitXorBuilder));
        self.builders.insert(
            AggregateExpressionType::Covariance,
            Box::new(CovarianceBuilder),
        );
        self.builders
            .insert(AggregateExpressionType::Variance, Box::new(VarianceBuilder));
        self.builders
            .insert(AggregateExpressionType::Stddev, Box::new(StddevBuilder));
        self.builders.insert(
            AggregateExpressionType::Correlation,
            Box::new(CorrelationBuilder),
        );
        self.builders.insert(
            AggregateExpressionType::BloomFilterAgg,
            Box::new(BloomFilterAggBuilder),
        );
    }

    /// Extract aggregate expression type from Spark protobuf aggregate expression
    fn get_aggregate_expression_type(
        spark_expr: &AggExpr,
    ) -> Result<AggregateExpressionType, ExecutionError> {
        match spark_expr.expr_struct.as_ref() {
            Some(AggExprStruct::Count(_)) => Ok(AggregateExpressionType::Count),
            Some(AggExprStruct::Min(_)) => Ok(AggregateExpressionType::Min),
            Some(AggExprStruct::Max(_)) => Ok(AggregateExpressionType::Max),
            Some(AggExprStruct::Sum(_)) => Ok(AggregateExpressionType::Sum),
            Some(AggExprStruct::Avg(_)) => Ok(AggregateExpressionType::Avg),
            Some(AggExprStruct::First(_)) => Ok(AggregateExpressionType::First),
            Some(AggExprStruct::Last(_)) => Ok(AggregateExpressionType::Last),
            Some(AggExprStruct::BitAndAgg(_)) => Ok(AggregateExpressionType::BitAndAgg),
            Some(AggExprStruct::BitOrAgg(_)) => Ok(AggregateExpressionType::BitOrAgg),
            Some(AggExprStruct::BitXorAgg(_)) => Ok(AggregateExpressionType::BitXorAgg),
            Some(AggExprStruct::Covariance(_)) => Ok(AggregateExpressionType::Covariance),
            Some(AggExprStruct::Variance(_)) => Ok(AggregateExpressionType::Variance),
            Some(AggExprStruct::Stddev(_)) => Ok(AggregateExpressionType::Stddev),
            Some(AggExprStruct::Correlation(_)) => Ok(AggregateExpressionType::Correlation),
            Some(AggExprStruct::BloomFilterAgg(_)) => Ok(AggregateExpressionType::BloomFilterAgg),
            Some(other) => Err(ExecutionError::GeneralError(format!(
                "Unsupported aggregate expression type: {:?}",
                other
            ))),
            None => internal_err!("Aggregate expression struct is None".to_string(),),
        }
    }
}

/// Register query context for an aggregate expression if present
fn register_query_context(
    spark_expr: &AggExpr,
    registry: &Arc<datafusion_comet_spark_expr::QueryContextMap>,
) {
    if let (Some(expr_id), Some(ctx_proto)) =
        (spark_expr.expr_id, spark_expr.query_context.as_ref())
    {
        // Deserialize QueryContext from protobuf
        let query_ctx = datafusion_comet_spark_expr::QueryContext::new(
            ctx_proto.sql_text.clone(),
            ctx_proto.start_index,
            ctx_proto.stop_index,
            ctx_proto.object_type.clone(),
            ctx_proto.object_name.clone(),
            ctx_proto.line,
            ctx_proto.start_position,
        );

        // Register query context for error reporting
        registry.register(expr_id, query_ctx);
    }
}
