CREATE EXTENSION IF NOT EXISTS pg_cron;

SELECT cron.schedule('compute_hot_or_not', '*/5 * * * *', 'SELECT hot_or_not_evaluator.compute_hot_or_not()');

SELECT * FROM cron.job;


## 

