# Databricks notebook source
# MAGIC %md
# MAGIC ### Create a query that checks the percentage of MAE being higher than 7000

# COMMAND ----------

import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

srcs = w.data_sources.list()


alert_query = """
SELECT 
  (COUNT(CASE WHEN mean_absolute_error > 70000 THEN 1 END) * 100.0 / COUNT(CASE WHEN mean_absolute_error IS NOT NULL AND NOT isnan(mean_absolute_error) THEN 1 END)) AS percentage_higher_than_70000
FROM mlops_dev.house_prices.model_monitoring_profile_metrics"""


query = w.queries.create(query=sql.CreateQueryRequestQuery(display_name=f'house-price-alert-query-{time.time_ns()}',
                                                           warehouse_id=srcs[0].warehouse_id,
                                                           description="Alert on house price model MAE",
                                                           query_text=alert_query))

alert = w.alerts.create(
    alert=sql.CreateAlertRequestAlert(condition=sql.AlertCondition(operand=sql.AlertConditionOperand(
        column=sql.AlertOperandColumn(name="percentage_higher_than_70000")),
            op=sql.AlertOperator.GREATER_THAN,
            threshold=sql.AlertConditionThreshold(
                value=sql.AlertOperandValue(
                    double_value=45))),
            display_name=f'house-price-mae-alert-{time.time_ns()}',
            query_id=query.id
        )
    )



# COMMAND ----------

# cleanup
w.queries.delete(id=query.id)
w.alerts.delete(id=alert.id)