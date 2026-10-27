"""Mature relational execution for Researcher data actions.

The Agent owns the typed request and provenance contract.  DuckDB owns the
relational semantics: filtering, projection, grouping, aggregation, joins,
and row materialisation.  This module intentionally contains no scientific
field aliases or research workflow decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import duckdb
import pandas as pd
from rapidfuzz import fuzz, process


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _frame(rows: Iterable[dict[str, Any]], columns: Iterable[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(str(column) for column in columns))
    materialized = [dict(row) for row in rows]
    if not materialized:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame.from_records(materialized)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, columns]


@dataclass(frozen=True)
class RelationTable:
    table_id: str
    alias: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]


class DuckDBRelationalEngine:
    """Execute a bounded relational request with DuckDB, not Python loops."""

    def __init__(self) -> None:
        self.connection = duckdb.connect(database=":memory:")

    def close(self) -> None:
        self.connection.close()

    def execute_table(self, rows: Iterable[dict[str, Any]], columns: Iterable[str], request: dict[str, Any]) -> dict[str, Any]:
        frame = _frame(rows, columns)
        self.connection.register("workspace_table", frame)
        try:
            operation = str(request.get("operation") or "")
            where, parameters = self._where(request.get("filters") or [])
            if operation in {"sample", "query"}:
                selected = [str(field) for field in request.get("columns", []) if field]
                projection = ", ".join(_quote(field) for field in selected) if selected else "*"
                limit = max(1, int(request.get("row_limit") or 100))
                records = self.connection.execute(
                    f"SELECT {projection} FROM workspace_table{where} LIMIT ?", [*parameters, limit]
                ).fetchdf().to_dict("records")
                count = self._count(where, parameters)
                return {"record_count": count, "returned_records": len(records), "records": records, "sample": records[:20], "truncated": count > len(records)}
            if operation == "profile":
                selected = [str(field) for field in request.get("columns", []) if field] or list(frame.columns)
                profile: dict[str, dict[str, int]] = {}
                for field in selected:
                    result = self.connection.execute(
                        f"SELECT COUNT(NULLIF(CAST({_quote(field)} AS VARCHAR), '')), COUNT(DISTINCT NULLIF(CAST({_quote(field)} AS VARCHAR), '')) FROM workspace_table{where}", parameters
                    ).fetchone()
                    profile[field] = {"non_null": int(result[0]), "distinct": int(result[1])}
                return {"record_count": self._count(where, parameters), "columns": profile}
            if operation == "aggregate":
                return self._aggregate(where, parameters, request)
            if operation == "materialize":
                return {"record_count": self._count(where, parameters)}
            raise ValueError(f"DuckDB table engine does not execute operation: {operation!r}")
        finally:
            self.connection.unregister("workspace_table")

    def relation_probe_or_count(self, tables: list[RelationTable], steps: list[dict[str, Any]], edges: list[dict[str, Any]], row_limit: int) -> tuple[list[dict[str, Any]], int, bool]:
        self._register_tables(tables)
        try:
            reports = [self._edge_report(edge, tables) for edge in edges]
            join_sql = self._join_sql(tables, steps, count_only=True)
            count = int(self.connection.execute(join_sql).fetchone()[0])
            return reports, count, count > row_limit
        finally:
            self._unregister_tables(tables)

    def fuzzy_match_pair(self, left: RelationTable, right: RelationTable, left_key: str, right_key: str, *, threshold: float, row_limit: int) -> dict[str, Any]:
        """Use Splink for an explicitly authorized fuzzy relation.

        This remains an implementation of the generic relation operation.
        The Researcher supplies the policy and threshold; RapidFuzz supplies
        the mature string-similarity calculation.  No fuzzy default exists.
        """

        left_rows, right_rows = [dict(row) for row in left.rows], [dict(row) for row in right.rows]
        if not left_rows or not right_rows:
            return {"provider": "rapidfuzz", "candidate_pair_count": 0, "accepted_pairs": [], "ambiguous_pairs": []}
        choices = {f"right_{index}": str(row.get(right_key) or "") for index, row in enumerate(right_rows)}
        review_threshold = max(0.0, threshold - 0.1)
        grouped: dict[str, list[dict[str, Any]]] = {}
        candidate_count = 0
        for index, row in enumerate(left_rows):
            left_id = f"left_{index}"
            value = str(row.get(left_key) or "")
            candidates = process.extract(value, choices, scorer=fuzz.WRatio, score_cutoff=review_threshold * 100, limit=2)
            grouped[left_id] = []
            for _, score, right_id in candidates:
                candidate_count += 1
                grouped[left_id].append({"left_id": left_id, "right_id": str(right_id), "match_probability": float(score) / 100.0, "match_weight": None})
        accepted, ambiguous = [], []
        for candidates in grouped.values():
            candidates.sort(key=lambda item: (-item["match_probability"], item["right_id"]))
            best = candidates[0]
            second = candidates[1]["match_probability"] if len(candidates) > 1 else 0.0
            best["candidate_count_for_left"] = len(candidates)
            best["score_margin"] = best["match_probability"] - second
            if best["match_probability"] >= threshold and best["score_margin"] >= 0.1:
                accepted.append(best)
            else:
                ambiguous.append(best)
        return {
            "provider": "rapidfuzz",
            "candidate_pair_count": candidate_count,
            "accepted_pairs": accepted[: max(1, int(row_limit))],
            "ambiguous_pairs": ambiguous[: max(1, int(row_limit))],
            "accepted_pair_count": len(accepted),
            "ambiguous_pair_count": len(ambiguous),
        }

    @staticmethod
    def load_fuzzy_pairs(left: RelationTable, right: RelationTable, pairs: list[dict[str, Any]], row_limit: int) -> list[dict[str, Any]]:
        left_rows = {f"left_{index}": row for index, row in enumerate(left.rows)}
        right_rows = {f"right_{index}": row for index, row in enumerate(right.rows)}
        result = []
        for pair in pairs[: max(1, int(row_limit))]:
            left_row, right_row = left_rows.get(pair.get("left_id")), right_rows.get(pair.get("right_id"))
            if left_row is None or right_row is None:
                continue
            result.append({
                **{f"left__{key}": value for key, value in left_row.items()},
                **{f"right__{key}": value for key, value in right_row.items()},
                "match_probability": pair.get("match_probability"),
                "match_weight": pair.get("match_weight"),
            })
        return result

    def load_relation(self, tables: list[RelationTable], steps: list[dict[str, Any]], row_limit: int | None = None) -> list[dict[str, Any]]:
        self._register_tables(tables)
        try:
            sql = self._join_sql(tables, steps, count_only=False)
            if row_limit is not None:
                sql += f" LIMIT {int(row_limit) + 1}"
            return self.connection.execute(sql).fetchdf().to_dict("records")
        finally:
            self._unregister_tables(tables)

    def load_pair(self, left: RelationTable, right: RelationTable, left_key: str, right_key: str, join_type: str, row_limit: int, match_policy: str = "exact") -> list[dict[str, Any]]:
        """Load a two-table relation, including semi/anti joins, with SQL."""
        self._register_tables([left, right])
        try:
            left_alias, right_alias = _quote(left.alias), _quote(right.alias)
            condition = self._match_condition(left_alias, left_key, right_alias, right_key, match_policy)
            if join_type in {"semi", "anti"}:
                existence = "EXISTS" if join_type == "semi" else "NOT EXISTS"
                sql = f"SELECT " + ", ".join(f"{left_alias}.{_quote(column)} AS {_quote(f'left__{column}') }" for column in left.columns)
                sql += f" FROM relation_0 AS {left_alias} WHERE {existence} (SELECT 1 FROM relation_1 AS {right_alias} WHERE {condition}) LIMIT {int(row_limit) + 1}"
            else:
                join = "LEFT JOIN" if join_type == "left" else "JOIN"
                projection = [f"{left_alias}.{_quote(column)} AS {_quote(f'left__{column}') }" for column in left.columns]
                projection.extend(f"{right_alias}.{_quote(column)} AS {_quote(f'right__{column}') }" for column in right.columns)
                sql = f"SELECT {', '.join(projection)} FROM relation_0 AS {left_alias} {join} relation_1 AS {right_alias} ON {condition} LIMIT {int(row_limit) + 1}"
            return self.connection.execute(sql).fetchdf().to_dict("records")
        finally:
            self._unregister_tables([left, right])

    def _count(self, where: str, parameters: list[Any]) -> int:
        return int(self.connection.execute(f"SELECT COUNT(*) FROM workspace_table{where}", parameters).fetchone()[0])

    def _aggregate(self, where: str, parameters: list[Any], request: dict[str, Any]) -> dict[str, Any]:
        group_by = [str(field) for field in request.get("group_by", []) if field]
        bin_field = str(request.get("bin_field") or "")
        width = float(request.get("bin_width") or 0)
        origin = float(request.get("bin_origin") or 0)
        expressions: list[str] = [_quote(field) for field in group_by]
        aliases: list[str] = list(group_by)
        original_parameters = list(parameters)
        if bin_field and width > 0:
            expressions.append(f"floor((CAST({_quote(bin_field)} AS DOUBLE) - ?) / ?) * ? + ?")
            # SELECT placeholders occur before WHERE placeholders in the SQL
            # statement.  Keep the binding order identical to that layout.
            parameters = [origin, width, width, origin, *parameters]
            aliases.append(f"{bin_field}_bin_start")
        else:
            parameters = list(parameters)
        # Every non-aggregate projection, including a bin expression, must be
        # grouped.  The old count was taken before appending the bin and
        # generated invalid DuckDB SQL for age/longitude bins.
        grouping_count = len(expressions)
        aggregate_specs = request.get("aggregates") or [{"field": "", "operation": "count", "as": "count"}]
        for spec in aggregate_specs:
            field = str(spec.get("field") or "")
            operation = str(spec.get("operation") or "count")
            alias = str(spec.get("as") or f"{operation}_{field}")
            if operation == "count":
                expression = f"COUNT({_quote(field)})" if field else "COUNT(*)"
            elif operation == "n_unique": expression = f"COUNT(DISTINCT {_quote(field)})"
            elif operation in {"min", "max", "mean", "median", "std"}:
                function = {"mean": "AVG", "median": "MEDIAN", "std": "STDDEV_POP"}.get(operation, operation.upper())
                expression = f"{function}(TRY_CAST({_quote(field)} AS DOUBLE))"
            else:
                raise ValueError(f"Unsupported DuckDB aggregate operation: {operation}")
            expressions.append(expression)
            aliases.append(alias)
        projection = ", ".join(f"{expr} AS {_quote(alias)}" for expr, alias in zip(expressions, aliases))
        group_clause = f" GROUP BY {', '.join(str(index + 1) for index in range(grouping_count))}" if grouping_count else ""
        limit = max(1, int(request.get("row_limit") or 500))
        query = f"SELECT {projection} FROM workspace_table{where}{group_clause} ORDER BY ALL LIMIT ?"
        rows = self.connection.execute(query, [*parameters, limit + 1]).fetchdf().to_dict("records")
        return {"input_record_count": self._count(where, original_parameters), "group_count": len(rows), "rows": rows[:limit], "truncated": len(rows) > limit}

    def _where(self, filters: list[dict[str, Any]]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for predicate in filters:
            field = str(predicate.get("field") or "")
            operator = str(predicate.get("operator") or "")
            column = _quote(field)
            if operator == "is_null":
                clauses.append(f"{column} IS NULL")
            elif operator == "not_null":
                clauses.append(f"{column} IS NOT NULL")
            elif operator == "between":
                clauses.append(f"TRY_CAST({column} AS DOUBLE) BETWEEN ? AND ?")
                parameters.extend([predicate.get("minimum_number"), predicate.get("maximum_number")])
            elif operator == "contains":
                values = [str(item) for item in predicate.get("text_values", [])]
                if not values:
                    clauses.append("FALSE")
                    continue
                clauses.append("(" + " OR ".join(f"lower(CAST({column} AS VARCHAR)) LIKE lower(CAST(? AS VARCHAR))" for _ in values) + ")")
                parameters.extend([f"%{value}%" for value in values])
            elif operator in {"eq", "in"}:
                values = [str(item) for item in predicate.get("text_values", [])]
                if not values:
                    clauses.append("FALSE")
                    continue
                clauses.append("lower(CAST({} AS VARCHAR)) IN ({})".format(column, ", ".join("lower(CAST(? AS VARCHAR))" for _ in values)))
                parameters.extend(values)
            else:
                raise ValueError(f"Unsupported filter operator: {operator!r}")
        return (f" WHERE {' AND '.join(clauses)}" if clauses else ""), parameters

    def _register_tables(self, tables: list[RelationTable]) -> None:
        for index, table in enumerate(tables):
            self.connection.register(f"relation_{index}", _frame(table.rows, table.columns))

    def _unregister_tables(self, tables: list[RelationTable]) -> None:
        for index, _ in enumerate(tables):
            self.connection.unregister(f"relation_{index}")

    def _edge_report(self, edge: dict[str, Any], tables: list[RelationTable]) -> dict[str, Any]:
        left, right = next(table for table in tables if table.table_id == edge["left_table_id"]), next(table for table in tables if table.table_id == edge["right_table_id"])
        left_index, right_index = tables.index(left), tables.index(right)
        lkey, rkey = _quote(edge["left_key"]), _quote(edge["right_key"])
        policy = str(edge.get("match_policy") or "exact")
        condition = self._match_condition("l", edge["left_key"], "r", edge["right_key"], policy)
        join = f"FROM relation_{left_index} l JOIN relation_{right_index} r ON {condition}"
        row = self.connection.execute(f"SELECT COUNT(DISTINCT CAST(l.{lkey} AS VARCHAR)), COUNT(*), COUNT(DISTINCT CAST(r.{rkey} AS VARCHAR)) {join} WHERE l.{lkey} IS NOT NULL AND r.{rkey} IS NOT NULL").fetchone()
        left_count = int(self.connection.execute(f"SELECT COUNT(*) FROM relation_{left_index}").fetchone()[0])
        right_count = int(self.connection.execute(f"SELECT COUNT(*) FROM relation_{right_index}").fetchone()[0])
        left_distinct = int(self.connection.execute(f"SELECT COUNT(DISTINCT CAST({lkey} AS VARCHAR)) FROM relation_{left_index} WHERE {lkey} IS NOT NULL").fetchone()[0])
        right_distinct = int(self.connection.execute(f"SELECT COUNT(DISTINCT CAST({rkey} AS VARCHAR)) FROM relation_{right_index} WHERE {rkey} IS NOT NULL").fetchone()[0])
        left_duplicates = int(self.connection.execute(f"SELECT COUNT(*) FROM (SELECT {lkey} FROM relation_{left_index} WHERE {lkey} IS NOT NULL GROUP BY {lkey} HAVING COUNT(*) > 1)").fetchone()[0])
        right_duplicates = int(self.connection.execute(f"SELECT COUNT(*) FROM (SELECT {rkey} FROM relation_{right_index} WHERE {rkey} IS NOT NULL GROUP BY {rkey} HAVING COUNT(*) > 1)").fetchone()[0])
        unmatched_left = int(self.connection.execute(f"SELECT COUNT(*) FROM relation_{left_index} l WHERE l.{lkey} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM relation_{right_index} r WHERE {condition})").fetchone()[0])
        unmatched_right = int(self.connection.execute(f"SELECT COUNT(*) FROM relation_{right_index} r WHERE r.{rkey} IS NOT NULL AND NOT EXISTS (SELECT 1 FROM relation_{left_index} l WHERE {condition})").fetchone()[0])
        return {**edge, "left_row_count": left_count, "right_row_count": right_count, "left_non_null_key_count": int(self.connection.execute(f"SELECT COUNT(*) FROM relation_{left_index} WHERE {lkey} IS NOT NULL").fetchone()[0]), "right_non_null_key_count": int(self.connection.execute(f"SELECT COUNT(*) FROM relation_{right_index} WHERE {rkey} IS NOT NULL").fetchone()[0]), "matched_distinct_keys": int(row[0]), "joined_pair_count": int(row[1]), "left_distinct_key_count": left_distinct, "right_distinct_key_count": right_distinct, "unmatched_left_row_count": unmatched_left, "unmatched_right_row_count": unmatched_right, "left_duplicate_key_count": left_duplicates, "right_duplicate_key_count": right_duplicates}

    def _join_sql(self, tables: list[RelationTable], steps: list[dict[str, Any]], *, count_only: bool) -> str:
        by_id = {table.table_id: (index, table) for index, table in enumerate(tables)}
        root = tables[0]
        from_clause = f"FROM relation_0 AS {_quote(root.alias)}"
        for step in steps:
            existing_index, existing = by_id[step["existing_table_id"]]
            new_index, new = by_id[step["new_table_id"]]
            join = "LEFT JOIN" if step.get("join_type") == "left" else "JOIN"
            condition = self._match_condition(_quote(existing.alias), step["existing_key"], _quote(new.alias), step["new_key"], str(step.get("match_policy") or "exact"))
            from_clause += f" {join} relation_{new_index} AS {_quote(new.alias)} ON {condition}"
        if count_only:
            return f"SELECT COUNT(*) {from_clause}"
        projection = []
        for table in tables:
            for column in table.columns:
                projection.append(f"{_quote(table.alias)}.{_quote(column)} AS {_quote(f'{table.alias}__{column}')}" )
        return f"SELECT {', '.join(projection)} {from_clause}"

    @staticmethod
    def _match_condition(left_alias: str, left_key: str, right_alias: str, right_key: str, match_policy: str) -> str:
        left = f"CAST({left_alias}.{_quote(left_key)} AS VARCHAR)"
        right = f"CAST({right_alias}.{_quote(right_key)} AS VARCHAR)"
        if match_policy == "exact":
            return f"{left} = {right}"
        if match_policy == "normalized":
            # This is an explicit, deterministic rule selected by Researcher;
            # the provider never upgrades exact matching to this policy.
            return f"lower(regexp_replace({left}, '\\s+', '', 'g')) = lower(regexp_replace({right}, '\\s+', '', 'g'))"
        raise ValueError(f"Relational engine has no deterministic provider for match policy: {match_policy!r}")
