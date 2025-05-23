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

import com.alibaba.cloud.ai.analyticdb.AnalyticDbVectorStoreProperties;
import com.alibaba.cloud.ai.request.SearchRequestDto;
import com.aliyun.gpdb20160503.Client;
import com.aliyun.gpdb20160503.models.*;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
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

	@Autowired
	@Qualifier("rawTextEmbeddingModel")
	private EmbeddingModel embeddingModel;

	@Autowired
	private MongoDbVectorStoreProperties mamgoDbVectorStoreProperties;

	@Autowired
	private MongoDBAtlasVectorStore vectorStore;


	/**
	 * 获取向量库中的文档
	 */
	public List<Document> getDocuments(String query, String vectorType) {
		SearchRequestDto request = new SearchRequestDto();
		request.setQuery(query);
		request.setVectorType(vectorType);
		request.setTopK(100);
		return searchWithVectorType(request);
	}

	/**
	 * 默认 filter 的搜索接口
	 */
	public List<Document> searchWithVectorType(SearchRequestDto searchRequestDTO) {
		String filter = String.format("metadata.vectorType = '%s'",
				searchRequestDTO.getVectorType());

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
			List<Document> response = vectorStore.doSimilaritySearch(vecRequest);
			return parseDocuments(response);
		} catch (Exception e) {
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
			MongoTemplate mongoTemplate = vectorStore.<MongoTemplate>getNativeClient().get();
			List<org.bson.Document> response = mongoTemplate.find(query, org.bson.Document.class,mamgoDbVectorStoreProperties.getCollectName());

			return parseBsonDocuments(response);
		} catch (Exception e) {
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
					} catch (Exception e) {
						throw new RuntimeException("解析元数据失败: " + e.getMessage(), e);
					}
				})
				.collect(Collectors.toList());
	}

	private List<Document> parseBsonDocuments(List<org.bson.Document> response) throws Exception {
		return response.stream()
				.filter(match -> match.get("score") == null || match.getDouble("score") > 0.1 || match.getDouble("score") == 0.0)
				.map(match -> {
					Map<String, String> metadata = match.get("metadata",Map.class);
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

}
