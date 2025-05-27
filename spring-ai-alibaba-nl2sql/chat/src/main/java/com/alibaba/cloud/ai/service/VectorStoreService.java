/*
 * Copyright 2024-2025 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.alibaba.cloud.ai.service;

import com.alibaba.cloud.ai.dbconnector.DbAccessor;
import com.alibaba.cloud.ai.dbconnector.DbConfig;
import com.alibaba.cloud.ai.dbconnector.bo.ColumnInfoBO;
import com.alibaba.cloud.ai.dbconnector.bo.DbQueryParameter;
import com.alibaba.cloud.ai.dbconnector.bo.ForeignKeyInfoBO;
import com.alibaba.cloud.ai.dbconnector.bo.TableInfoBO;
import com.alibaba.cloud.ai.request.DeleteRequest;
import com.alibaba.cloud.ai.request.EvidenceRequest;
import com.alibaba.cloud.ai.request.SchemaInitRequest;
import com.alibaba.cloud.ai.request.SearchRequestDto;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.filter.Filter;
import org.springframework.ai.vectorstore.mongodb.ignite.MongoDBAtlasFilterExpressionConverter;
import org.springframework.ai.vectorstore.mongodb.ignite.MongoDBAtlasVectorStore;
import org.springframework.ai.vectorstore.mongodb.ignite.MongoDbVectorStoreProperties;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Criteria;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Service
public class VectorStoreService {

	private static final String CONTENT_FIELD_NAME = "content";

	private static final String METADATA_FIELD_NAME = "metadata";

	private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

	private HashMap<String, MongoDBAtlasVectorStore> storeMap = new HashMap<>();

	@Autowired
	@Qualifier("rawTextEmbeddingModel")
	private EmbeddingModel embeddingModel;

	@Autowired
	private MongoTemplate mongoTemplate;

	@Autowired
	private MongoDbVectorStoreProperties mangoDbVectorStoreProperties;

	@Autowired
	private MongoDBAtlasVectorStore defaultVectorStore;

	@Autowired
	private DbAccessor dbAccessor;

	private Map<String, List<String>> foreignKeyMap;

	private int columnSamples = 0;

	public MongoDBAtlasVectorStore getByName(String name) {
		MongoDBAtlasVectorStore vectorStore = storeMap.get(name);
		if (vectorStore != null) {
			return vectorStore;
		}

		vectorStore = MongoDBAtlasVectorStore.builder(mongoTemplate, embeddingModel)
			.collectionName("vector_store_" + name)
			.filterExpressionConverter(new MongoDBAtlasFilterExpressionConverter())
			.initializeSchema(true)
			.build();

		try {
			vectorStore.afterPropertiesSet();
		}
		catch (Exception e) {
			throw new RuntimeException(e);
		}

		storeMap.put(name, vectorStore);

		return vectorStore;
	}

	/**
	 * 获取向量库中的文档
	 */
	public List<Document> getDocuments(String query, String vectorType) {
		return getDocuments(query, vectorType, 20);
	}

	public List<Document> getDocuments(String query, String vectorType, int topk) {
		SearchRequestDto request = new SearchRequestDto();
		request.setQuery(query);
		request.setVectorType(vectorType);
		request.setTopK(topk);
		return searchWithVectorType(request);
	}

	/**
	 * 默认 filter 的搜索接口
	 */
	public List<Document> searchWithVectorType(SearchRequestDto searchRequestDTO) {
		String filter = String.format("metadata.vectorType = '%s'", searchRequestDTO.getVectorType());

		searchRequestDTO.setFilterFormatted(filter);

		return executeQuery(searchRequestDTO);
	}

	/**
	 * 自定义 filter 的搜索接口
	 */
	public List<Document> searchWithFilter(SearchRequestDto searchRequestDTO) {
		searchRequestDTO.setFilterFormatted(searchRequestDTO.getFilterFormatted());
		return executeFilter(searchRequestDTO);
	}

	/**
	 * 执行实际查询并解析结果
	 */
	private List<Document> executeQuery(SearchRequestDto request) {
		try {
			SearchRequest vecRequest = SearchRequest.builder()
				.query(request.getQuery())
				.topK(request.getTopK())
				.build();
			MongoDBAtlasVectorStore vectorStore = getByName(request.getVectorType());
			List<Document> response = vectorStore.doSimilaritySearch(vecRequest);
			return parseDocuments(response);
		}
		catch (Exception e) {
			throw new RuntimeException("向量数据库查询失败: " + e.getMessage(), e);
		}
	}

	/**
	 * 执行实际查询并解析结果
	 */
	private List<Document> executeFilter(SearchRequestDto request) {
		try {
			Query query = Query.query(Criteria.where("text").is(request.getVectorType()));
			query.limit(request.getTopK());
			MongoDBAtlasVectorStore vectorStore = getByName(request.getVectorType());
			MongoTemplate mongoTemplate = vectorStore.<MongoTemplate>getNativeClient().get();
			List<org.bson.Document> response = mongoTemplate.find(query, org.bson.Document.class,
					"vector_store_" + request.getVectorType());

			return parseBsonDocuments(response);
		}
		catch (Exception e) {
			throw new RuntimeException("向量数据库查询失败: " + e.getMessage(), e);
		}
	}

	/**
	 * 解析响应数据为 Document 列表
	 */
	private List<Document> parseDocuments(List<Document> response) throws Exception {
		return response.stream()
			.filter(match -> match.getScore() == null || match.getScore() > 0.1 || match.getScore() == 0.0)
			.map(match -> {

				try {
					Map<String, Object> metadata = match.getMetadata();

					return match;
				}
				catch (Exception e) {
					throw new RuntimeException("解析元数据失败: " + e.getMessage(), e);
				}
			})
			.collect(Collectors.toList());
	}

	private List<Document> parseBsonDocuments(List<org.bson.Document> response) throws Exception {
		return response.stream()
			.filter(match -> match.get("score") == null || match.getDouble("score") > 0.1
					|| match.getDouble("score") == 0.0)
			.map(match -> {
				Map<String, String> metadata = match.get("metadata", Map.class);
				try {
					Map<String, Object> metadataJson = OBJECT_MAPPER.readValue(metadata.get(METADATA_FIELD_NAME),
							new TypeReference<HashMap<String, Object>>() {
							});
					metadataJson.put("score", match.getDouble("score"));

					return new Document(match.get("_id").toString(), metadata.get(CONTENT_FIELD_NAME), metadataJson);
				}
				catch (Exception e) {
					throw new RuntimeException("解析元数据失败: " + e.getMessage(), e);
				}
			})
			.collect(Collectors.toList());
	}

	/**
	 * 将证据内容添加到向量库中
	 * @param evidenceRequests 证据请求列表
	 * @return 是否成功
	 */
	public Boolean addEvidence(List<EvidenceRequest> evidenceRequests) {
		List<Document> evidences = new ArrayList<>();
		for (EvidenceRequest req : evidenceRequests) {
			Document doc = new Document(UUID.randomUUID().toString(), req.getContent(),
					Map.of("evidenceType", req.getType(), "vectorType", "evidence"));
			evidences.add(doc);
		}
		MongoDBAtlasVectorStore vectorStore = getByName("evidence");
		vectorStore.add(evidences);
		return true;
	}

	/**
	 * 将文本嵌入为向量
	 * @param text 输入文本
	 * @return 向量化结果
	 */
	public List<Double> embed(String text) {
		float[] embedded = embeddingModel.embed(text);
		List<Double> result = new ArrayList<>();
		for (float value : embedded) {
			result.add((double) value);
		}
		return result;
	}

	/**
	 * 删除指定条件的向量数据
	 * @param deleteRequest 删除请求
	 * @return 是否删除成功
	 */
	public Boolean deleteDocuments(DeleteRequest deleteRequest) throws Exception {
		try {
			MongoDBAtlasVectorStore vectorStore = getByName(deleteRequest.getVectorType());
			String filterExpression;
			if (deleteRequest.getId() != null && !deleteRequest.getId().isEmpty()) {
				filterExpression = String.format("id = '%s'", deleteRequest.getId());

				vectorStore.doDelete(List.of(deleteRequest.getId()));
			}
			else if (deleteRequest.getVectorType() != null && !deleteRequest.getVectorType().isEmpty()) {
				filterExpression = String.format("jsonb_extract_path_text(metadata, 'vectorType') = '%s'",
						deleteRequest.getVectorType());
				Filter.Expression expression = new Filter.Expression(Filter.ExpressionType.EQ,
						new Filter.Key("vectorType"), new Filter.Value(deleteRequest.getVectorType()));
				vectorStore.doDelete(expression);
			}
			else {
				throw new IllegalArgumentException("Either id or vectorType must be specified.");
			}

			return true;
		}
		catch (Exception e) {
			throw new Exception("Failed to delete collection data by filterExpression: " + e.getMessage(), e);
		}
	}

	/**
	 * 初始化数据库 schema 到向量库
	 * @param schemaInitRequest schema 初始化请求
	 * @throws Exception 如果发生错误
	 */
	public Boolean schema(SchemaInitRequest schemaInitRequest) throws Exception {
		DbConfig dbConfig = schemaInitRequest.getDbConfig();
		DbQueryParameter dqp = DbQueryParameter.from(dbConfig)
			.setSchema(dbConfig.getSchema())
			.setTables(schemaInitRequest.getTables());

		DeleteRequest deleteRequest = new DeleteRequest();
		deleteRequest.setVectorType("column");
		deleteDocuments(deleteRequest);
		deleteRequest.setVectorType("table");
		deleteDocuments(deleteRequest);

		if (schemaInitRequest.getTables() == null) {
			List<TableInfoBO> tables = dbAccessor.showTables(dbConfig, dqp);
			dqp.setTables(tables.stream().map(t -> t.getName()).collect(Collectors.toList()));
		}

		List<ForeignKeyInfoBO> foreignKeyInfoBOS = dbAccessor.showForeignKeys(dbConfig, dqp);
		this.foreignKeyMap = buildForeignKeyMap(foreignKeyInfoBOS);

		List<TableInfoBO> tableInfoBOS = dbAccessor.fetchTables(dbConfig, dqp);
		for (TableInfoBO tableInfoBO : tableInfoBOS) {
			processTable(tableInfoBO, dqp, dbConfig);
		}

		List<Document> columnDocuments = tableInfoBOS.stream().flatMap(table -> {
			try {
				dqp.setTable(table.getName());
				return dbAccessor.showColumns(dbConfig, dqp).stream().map(columnInfoBO -> {
					dqp.setColumn(columnInfoBO.getName());
					columnInfoBO.setTableName(table.getName());
					if (columnSamples > 0) {
						try {
							List<String> sampleColumn = dbAccessor.sampleColumn(dbConfig, dqp);
							sampleColumn = Optional.ofNullable(sampleColumn)
								.orElse(new ArrayList<>())
								.stream()
								.filter(Objects::nonNull)
								.distinct()
								.filter(s -> s.length() <= 100)
								.limit(columnSamples)
								.toList();
							columnInfoBO.setSamples(OBJECT_MAPPER.writeValueAsString(sampleColumn));
						}
						catch (Exception e) {
							e.printStackTrace();
						}
					}
					else {
						columnInfoBO.setSamples("[]");
					}
					return convertToDocument(table, columnInfoBO);
				});
			}
			catch (Exception e) {
				throw new RuntimeException(e);
			}
		}).collect(Collectors.toList());

		MongoDBAtlasVectorStore vectorStore = getByName("column");

		vectorStore.add(columnDocuments);

		List<Document> tableDocuments = tableInfoBOS.stream()
			.map(this::convertTableToDocument)
			.collect(Collectors.toList());

		MongoDBAtlasVectorStore tableVectorStore = getByName("table");
		tableVectorStore.add(tableDocuments);

		return true;
	}

	private void processTable(TableInfoBO tableInfoBO, DbQueryParameter dqp, DbConfig dbConfig) throws Exception {
		dqp.setTable(tableInfoBO.getName());
		List<ColumnInfoBO> columnInfoBOS = dbAccessor.showColumns(dbConfig, dqp);

		ColumnInfoBO primaryColumnDO = columnInfoBOS.stream()
			.filter(ColumnInfoBO::isPrimary)
			.findFirst()
			.orElse(new ColumnInfoBO());

		tableInfoBO.setPrimaryKey(primaryColumnDO.getName());
		tableInfoBO.setForeignKey(String.join("、", buildForeignKeyList(tableInfoBO.getName())));
	}

	private Map<String, List<String>> buildForeignKeyMap(List<ForeignKeyInfoBO> foreignKeyInfoBOS) {
		Map<String, List<String>> foreignKeyMap = new HashMap<>();
		for (ForeignKeyInfoBO fk : foreignKeyInfoBOS) {
			String key = fk.getTable() + "." + fk.getColumn() + "=" + fk.getReferencedTable() + "."
					+ fk.getReferencedColumn();

			foreignKeyMap.computeIfAbsent(fk.getTable(), k -> new ArrayList<>()).add(key);
			foreignKeyMap.computeIfAbsent(fk.getReferencedTable(), k -> new ArrayList<>()).add(key);
		}
		return foreignKeyMap;
	}

	private List<String> buildForeignKeyList(String tableName) {
		List<String> foreignKey = this.foreignKeyMap.get(tableName);
		if (foreignKey != null) {
			return foreignKey;
		}
		return new ArrayList<>();
	}

	public Document convertToDocument(TableInfoBO tableInfoBO, ColumnInfoBO columnInfoBO) {
		String text = Optional.ofNullable(columnInfoBO.getDescription()).orElse(columnInfoBO.getName());
		Map<String, Object> metadata = Map.of("name", columnInfoBO.getName(), "tableName", tableInfoBO.getName(),
				"description", Optional.ofNullable(columnInfoBO.getDescription()).orElse(""), "type",
				columnInfoBO.getType(), "primary", columnInfoBO.isPrimary(), "notnull", columnInfoBO.isNotnull(),
				"samples", columnInfoBO.getSamples(), "vectorType", "column"); //
		return new Document(tableInfoBO.getName() + "." + columnInfoBO.getName(), text, metadata);
	}

	public Document convertTableToDocument(TableInfoBO tableInfoBO) {
		String text = Optional.ofNullable(tableInfoBO.getDescription()).orElse(tableInfoBO.getName());
		String schema = Optional.ofNullable(tableInfoBO.getSchema()).orElse("");
		Map<String, Object> metadata = Map.of("schema", schema, "name", tableInfoBO.getName(), "description",
				Optional.ofNullable(tableInfoBO.getDescription()).orElse(""), "foreignKey", tableInfoBO.getForeignKey(),
				"primaryKey", tableInfoBO.getPrimaryKey(), "vectorType", "table");
		return new Document((schema.isEmpty() ? "" : schema + ".") + tableInfoBO.getName(), text, metadata);
	}

}
