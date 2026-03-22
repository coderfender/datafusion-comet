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

//! Aggregate expression builders

use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion::functions_aggregate::bit_and_or_xor::{bit_and_udaf, bit_or_udaf, bit_xor_udaf};
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::functions_aggregate::first_last::{FirstValue, LastValue};
use datafusion::functions_aggregate::min_max::{max_udaf, min_udaf};
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::logical_expr::AggregateUDF;
use datafusion::physical_expr::aggregate::{AggregateExprBuilder, AggregateFunctionExpr};
use datafusion::physical_expr::expressions::{CastExpr, StatsType};
use datafusion::physical_expr::PhysicalExpr;
use datafusion_comet_proto::spark_expression::{agg_expr::ExprStruct as AggExprStruct, AggExpr};
use datafusion_comet_spark_expr::{
    Avg, AvgDecimal, BloomFilterAgg, Correlation, Covariance, Stddev, SumDecimal, SumInteger,
    Variance,
};

use crate::execution::operators::ExecutionError;
use crate::execution::operators::ExecutionError::GeneralError;
use crate::execution::planner::aggregate_registry::AggregateExpressionBuilder;
use crate::execution::planner::{from_protobuf_eval_mode, PhysicalPlanner};
use crate::execution::serde::to_arrow_datatype;

/// Macro to extract aggregate expression from AggExpr
#[macro_export]
macro_rules! extract_agg_expr {
    ($spark_expr:expr, $variant:ident) => {{
        match $spark_expr.expr_struct.as_ref() {
            Some(AggExprStruct::$variant(expr)) => expr,
            _ => {
                return Err(ExecutionError::GeneralError(format!(
                    "Expected {} expression",
                    stringify!($variant)
                )))
            }
        }
    }};
}

/// Helper function to create aggregate function expression with common parameters
fn create_aggr_func_expr(
    name: &str,
    schema: SchemaRef,
    children: Vec<Arc<dyn PhysicalExpr>>,
    func: AggregateUDF,
) -> Result<AggregateFunctionExpr, ExecutionError> {
    AggregateExprBuilder::new(Arc::new(func), children)
        .schema(schema)
        .alias(name)
        .with_ignore_nulls(false)
        .with_distinct(false)
        .build()
        .map_err(|e| e.into())
}

// ============================================================================
// Count Builder
// ============================================================================

pub struct CountBuilder;

impl AggregateExpressionBuilder for CountBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Count);
        assert!(!expr.children.is_empty());
        let children = expr
            .children
            .iter()
            .map(|child| planner.create_expr(child, Arc::clone(&schema)))
            .collect::<Result<Vec<_>, _>>()?;

        AggregateExprBuilder::new(count_udaf(), children)
            .schema(schema)
            .alias("count")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| ExecutionError::DataFusionError(e.to_string()))
    }
}

// ============================================================================
// Min Builder
// ============================================================================

pub struct MinBuilder;

impl AggregateExpressionBuilder for MinBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Min);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());
        let child = Arc::new(CastExpr::new(child, datatype.clone(), None));

        AggregateExprBuilder::new(min_udaf(), vec![child])
            .schema(schema)
            .alias("min")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| ExecutionError::DataFusionError(e.to_string()))
    }
}

// ============================================================================
// Max Builder
// ============================================================================

pub struct MaxBuilder;

impl AggregateExpressionBuilder for MaxBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Max);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());
        let child = Arc::new(CastExpr::new(child, datatype.clone(), None));

        AggregateExprBuilder::new(max_udaf(), vec![child])
            .schema(schema)
            .alias("max")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| ExecutionError::DataFusionError(e.to_string()))
    }
}

// ============================================================================
// Sum Builder
// ============================================================================

pub struct SumBuilder;

impl AggregateExpressionBuilder for SumBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Sum);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());

        let builder = match datatype {
            DataType::Decimal128(_, _) => {
                let eval_mode = from_protobuf_eval_mode(expr.eval_mode)?;
                let func = AggregateUDF::new_from_impl(SumDecimal::try_new(
                    datatype,
                    eval_mode,
                    spark_expr.expr_id,
                    Arc::clone(planner.query_context_registry()),
                )?);
                AggregateExprBuilder::new(Arc::new(func), vec![child])
            }
            DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
                let eval_mode = from_protobuf_eval_mode(expr.eval_mode)?;
                let func = AggregateUDF::new_from_impl(SumInteger::try_new(datatype, eval_mode)?);
                AggregateExprBuilder::new(Arc::new(func), vec![child])
            }
            _ => {
                // cast to the result data type of SUM if necessary, we should not expect
                // a cast failure since it should have already been checked at Spark side
                let child = Arc::new(CastExpr::new(Arc::clone(&child), datatype.clone(), None));
                AggregateExprBuilder::new(sum_udaf(), vec![child])
            }
        };
        builder
            .schema(schema)
            .alias("sum")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| e.into())
    }
}

// ============================================================================
// Avg Builder
// ============================================================================

pub struct AvgBuilder;

impl AggregateExpressionBuilder for AvgBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Avg);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());
        let input_datatype = to_arrow_datatype(expr.sum_datatype.as_ref().unwrap());

        let builder = match datatype {
            DataType::Decimal128(_, _) => {
                let eval_mode = from_protobuf_eval_mode(expr.eval_mode)?;
                let func = AggregateUDF::new_from_impl(AvgDecimal::new(
                    datatype,
                    input_datatype,
                    eval_mode,
                    spark_expr.expr_id,
                    Arc::clone(planner.query_context_registry()),
                ));
                AggregateExprBuilder::new(Arc::new(func), vec![child])
            }
            _ => {
                // For all other numeric types (Int8/16/32/64, Float32/64):
                // Cast to Float64 for accumulation
                let child: Arc<dyn PhysicalExpr> =
                    Arc::new(CastExpr::new(Arc::clone(&child), DataType::Float64, None));
                let func = AggregateUDF::new_from_impl(Avg::new("avg", DataType::Float64));
                AggregateExprBuilder::new(Arc::new(func), vec![child])
            }
        };
        builder
            .schema(schema)
            .alias("avg")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| e.into())
    }
}

// ============================================================================
// First Builder
// ============================================================================

pub struct FirstBuilder;

impl AggregateExpressionBuilder for FirstBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, First);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let func = AggregateUDF::new_from_impl(FirstValue::new());

        AggregateExprBuilder::new(Arc::new(func), vec![child])
            .schema(schema)
            .alias("first")
            .with_ignore_nulls(expr.ignore_nulls)
            .with_distinct(false)
            .build()
            .map_err(|e| e.into())
    }
}

// ============================================================================
// Last Builder
// ============================================================================

pub struct LastBuilder;

impl AggregateExpressionBuilder for LastBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Last);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let func = AggregateUDF::new_from_impl(LastValue::new());

        AggregateExprBuilder::new(Arc::new(func), vec![child])
            .schema(schema)
            .alias("last")
            .with_ignore_nulls(expr.ignore_nulls)
            .with_distinct(false)
            .build()
            .map_err(|e| e.into())
    }
}

// ============================================================================
// BitAnd Builder
// ============================================================================

pub struct BitAndBuilder;

impl AggregateExpressionBuilder for BitAndBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, BitAndAgg);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;

        AggregateExprBuilder::new(bit_and_udaf(), vec![child])
            .schema(schema)
            .alias("bit_and")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| e.into())
    }
}

// ============================================================================
// BitOr Builder
// ============================================================================

pub struct BitOrBuilder;

impl AggregateExpressionBuilder for BitOrBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, BitOrAgg);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;

        AggregateExprBuilder::new(bit_or_udaf(), vec![child])
            .schema(schema)
            .alias("bit_or")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| e.into())
    }
}

// ============================================================================
// BitXor Builder
// ============================================================================

pub struct BitXorBuilder;

impl AggregateExpressionBuilder for BitXorBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, BitXorAgg);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;

        AggregateExprBuilder::new(bit_xor_udaf(), vec![child])
            .schema(schema)
            .alias("bit_xor")
            .with_ignore_nulls(false)
            .with_distinct(false)
            .build()
            .map_err(|e| e.into())
    }
}

// ============================================================================
// Covariance Builder
// ============================================================================

pub struct CovarianceBuilder;

impl AggregateExpressionBuilder for CovarianceBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Covariance);
        let child1 = planner.create_expr(expr.child1.as_ref().unwrap(), Arc::clone(&schema))?;
        let child2 = planner.create_expr(expr.child2.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());

        match expr.stats_type {
            0 => {
                let func = AggregateUDF::new_from_impl(Covariance::new(
                    "covariance",
                    datatype,
                    StatsType::Sample,
                    expr.null_on_divide_by_zero,
                ));

                create_aggr_func_expr("covariance", schema, vec![child1, child2], func)
            }
            1 => {
                let func = AggregateUDF::new_from_impl(Covariance::new(
                    "covariance_pop",
                    datatype,
                    StatsType::Population,
                    expr.null_on_divide_by_zero,
                ));

                create_aggr_func_expr("covariance_pop", schema, vec![child1, child2], func)
            }
            stats_type => Err(GeneralError(format!(
                "Unknown StatisticsType {stats_type:?} for Covariance"
            ))),
        }
    }
}

// ============================================================================
// Variance Builder
// ============================================================================

pub struct VarianceBuilder;

impl AggregateExpressionBuilder for VarianceBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Variance);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());

        match expr.stats_type {
            0 => {
                let func = AggregateUDF::new_from_impl(Variance::new(
                    "variance",
                    datatype,
                    StatsType::Sample,
                    expr.null_on_divide_by_zero,
                ));

                create_aggr_func_expr("variance", schema, vec![child], func)
            }
            1 => {
                let func = AggregateUDF::new_from_impl(Variance::new(
                    "variance_pop",
                    datatype,
                    StatsType::Population,
                    expr.null_on_divide_by_zero,
                ));

                create_aggr_func_expr("variance_pop", schema, vec![child], func)
            }
            stats_type => Err(GeneralError(format!(
                "Unknown StatisticsType {stats_type:?} for Variance"
            ))),
        }
    }
}

// ============================================================================
// Stddev Builder
// ============================================================================

pub struct StddevBuilder;

impl AggregateExpressionBuilder for StddevBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Stddev);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());

        match expr.stats_type {
            0 => {
                let func = AggregateUDF::new_from_impl(Stddev::new(
                    "stddev",
                    datatype,
                    StatsType::Sample,
                    expr.null_on_divide_by_zero,
                ));

                create_aggr_func_expr("stddev", schema, vec![child], func)
            }
            1 => {
                let func = AggregateUDF::new_from_impl(Stddev::new(
                    "stddev_pop",
                    datatype,
                    StatsType::Population,
                    expr.null_on_divide_by_zero,
                ));

                create_aggr_func_expr("stddev_pop", schema, vec![child], func)
            }
            stats_type => Err(GeneralError(format!(
                "Unknown StatisticsType {stats_type:?} for Stddev"
            ))),
        }
    }
}

// ============================================================================
// Correlation Builder
// ============================================================================

pub struct CorrelationBuilder;

impl AggregateExpressionBuilder for CorrelationBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, Correlation);
        let child1 = planner.create_expr(expr.child1.as_ref().unwrap(), Arc::clone(&schema))?;
        let child2 = planner.create_expr(expr.child2.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());

        let func = AggregateUDF::new_from_impl(Correlation::new(
            "correlation",
            datatype,
            expr.null_on_divide_by_zero,
        ));

        create_aggr_func_expr("correlation", schema, vec![child1, child2], func)
    }
}

// ============================================================================
// BloomFilterAgg Builder
// ============================================================================

pub struct BloomFilterAggBuilder;

impl AggregateExpressionBuilder for BloomFilterAggBuilder {
    fn build(
        &self,
        spark_expr: &AggExpr,
        schema: SchemaRef,
        planner: &PhysicalPlanner,
    ) -> Result<AggregateFunctionExpr, ExecutionError> {
        let expr = extract_agg_expr!(spark_expr, BloomFilterAgg);
        let child = planner.create_expr(expr.child.as_ref().unwrap(), Arc::clone(&schema))?;
        let num_items =
            planner.create_expr(expr.num_items.as_ref().unwrap(), Arc::clone(&schema))?;
        let num_bits = planner.create_expr(expr.num_bits.as_ref().unwrap(), Arc::clone(&schema))?;
        let datatype = to_arrow_datatype(expr.datatype.as_ref().unwrap());

        let func = AggregateUDF::new_from_impl(BloomFilterAgg::new(
            Arc::clone(&num_items),
            Arc::clone(&num_bits),
            datatype,
        ));

        create_aggr_func_expr("bloom_filter_agg", schema, vec![child], func)
    }
}
