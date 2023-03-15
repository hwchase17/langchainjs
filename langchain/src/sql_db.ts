import type { DataSource as DataSourceT, DataSourceOptions } from "typeorm";
import {
  generateTableInfoFromTables,
  getTableAndColumnsName,
  SerializedSqlDatabase,
  SqlDatabaseDataSourceParams,
  SqlDatabaseOptionsParams,
  SqlTable,
  verifyIgnoreTablesExistInDatabase,
  verifyIncludeTablesExistInDatabase,
  verifyListTablesExistInDatabase,
} from "./util/sql_utils.js";

export class SqlDatabase
  implements SqlDatabaseOptionsParams, SqlDatabaseDataSourceParams
{
  appDataSourceOptions: DataSourceOptions;

  appDataSource: DataSourceT;

  allTables: Array<SqlTable> = [];

  includesTables: Array<string> = [];

  ignoreTables: Array<string> = [];

  sampleRowsInTableInfo = 3;

  protected constructor(fields: SqlDatabaseDataSourceParams) {
    this.appDataSource = fields.appDataSource;
    this.appDataSourceOptions = fields.appDataSource.options;
    if (fields?.includesTables && fields?.ignoreTables) {
      throw new Error("Cannot specify both include_tables and ignoreTables");
    }
    this.includesTables = fields?.includesTables ?? [];
    this.ignoreTables = fields?.ignoreTables ?? [];
    this.sampleRowsInTableInfo =
      fields?.sampleRowsInTableInfo ?? this.sampleRowsInTableInfo;
  }

  static async fromDataSourceParams(
    fields: SqlDatabaseDataSourceParams
  ): Promise<SqlDatabase> {
    const sqlDatabase = new SqlDatabase(fields);
    if (!sqlDatabase.appDataSource.isInitialized) {
      await sqlDatabase.appDataSource.initialize();
    }
    sqlDatabase.allTables = await getTableAndColumnsName(
      sqlDatabase.appDataSource
    );
    verifyIncludeTablesExistInDatabase(
      sqlDatabase.allTables,
      sqlDatabase.includesTables
    );
    verifyIgnoreTablesExistInDatabase(
      sqlDatabase.allTables,
      sqlDatabase.ignoreTables
    );
    return sqlDatabase;
  }

  static async fromOptionsParams(
    fields: SqlDatabaseOptionsParams
  ): Promise<SqlDatabase> {
    const { DataSource } = await import("typeorm");
    const dataSource = new DataSource(fields.appDataSourceOptions);
    return SqlDatabase.fromDataSourceParams({
      ...fields,
      appDataSource: dataSource,
    });
  }

  /**
   * Get information about specified tables.
   *
   * Follows best practices as specified in: Rajkumar et al, 2022
   * (https://arxiv.org/abs/2204.00498)
   *
   * If `sample_rows_in_table_info`, the specified number of sample rows will be
   * appended to each table description. This can increase performance as
   * demonstrated in the paper.
   */
  async getTableInfo(targetTables?: Array<string>): Promise<string> {
    let selectedTables = this.allTables;
    if (targetTables && targetTables.length > 0) {
      verifyListTablesExistInDatabase(
        this.allTables,
        targetTables,
        "Wrong target table name:"
      );
      selectedTables = this.allTables.filter((currentTable) =>
        targetTables.includes(currentTable.tableName)
      );
    }

    return generateTableInfoFromTables(
      selectedTables,
      this.appDataSource,
      this.sampleRowsInTableInfo
    );
  }

  /**
   * Execute a SQL command and return a string representing the results.
   * If the statement returns rows, a string of the results is returned.
   * If the statement returns no rows, an empty string is returned.
   */
  async run(command: string, fetch: "all" | "one" = "all"): Promise<string> {
    // TODO: Potential security issue here
    const res = await this.appDataSource.query(command);

    if (fetch === "all") {
      return JSON.stringify(res);
    }

    if (res?.length > 0) {
      return JSON.stringify(res[0]);
    }

    return "";
  }

  serialize(): SerializedSqlDatabase {
    return {
      _type: "sql_database",
      appDataSourceOptions: this.appDataSourceOptions,
      includesTables: this.includesTables,
      ignoreTables: this.ignoreTables,
      sampleRowsInTableInfo: this.sampleRowsInTableInfo,
    };
  }
}
