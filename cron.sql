CREATE EXTENSION IF NOT EXISTS pg_cron;

SELECT cron.schedule('compute_hot_or_not', '*/1 * * * *', 'SELECT hot_or_not_evaluator.compute_hot_or_not()');

SELECT cron.schedule('update_avg_reference_predicted_ds_score', '0 0 */3 * *', 'SELECT hot_or_not_evaluator.update_avg_reference_predicted_ds_score()');

SELECT * FROM cron.job;
