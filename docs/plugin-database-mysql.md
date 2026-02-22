# Plugin: plugin_database_mysql

MySQL database tool. Gives the LLM read/write access to a local MySQL database.

## Tool provided

`db_query(sql: str) → str` — executes any SQL statement and returns formatted results.

## Gate behavior

Per-table gates with read/write distinction. Control via `!autoAIdb`:

```
!autoAIdb status                  show current settings
!autoAIdb person read true        auto-allow SELECT on person table
!autoAIdb read true               auto-allow ALL table reads (wildcard default)
!autoAIdb write false             require gate for all writes (wildcard default)
```

`__meta__` controls SHOW TABLES / DESCRIBE queries.

## Dependencies

```bash
pip install mysql-connector-python>=8.0
```

## Environment variables

```
MYSQL_USER=<username>
MYSQL_PASS=<password>
```

Database name is `mymcp` on `localhost` (hardcoded default; change in `config.py` if needed).

## Enable

```bash
python agentctl.py enable plugin_database_mysql
```
