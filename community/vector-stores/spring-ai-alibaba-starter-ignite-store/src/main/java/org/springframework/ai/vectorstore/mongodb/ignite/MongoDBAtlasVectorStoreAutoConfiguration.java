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
package org.springframework.ai.vectorstore.mongodb.ignite;

import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.embedding.BatchingStrategy;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.ai.vectorstore.observation.VectorStoreObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.data.mongodb.core.MongoTemplate;

/**
 * @author HeYQ
 * @since 2025-03-04 13:10
 */
@AutoConfiguration
@ConditionalOnClass({ MongoTemplate.class, RawTextEmbeddingModel.class, MongoDBAtlasVectorStore.class })
@EnableConfigurationProperties({ MongoDbVectorStoreProperties.class })
@ConditionalOnProperty(prefix = "spring.ai.vectorstore.mongodb", name = "enabled", havingValue = "true",
		matchIfMissing = true)
public class MongoDBAtlasVectorStoreAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean(BatchingStrategy.class)
	public BatchingStrategy tokenCountBatchingStrategy() {
		return new TokenCountBatchingStrategy();
	}

	@Bean
	public RawTextEmbeddingModel rawTextEmbeddingModel() {
		return new RawTextEmbeddingModel();
	}

	@Bean
	@ConditionalOnMissingBean
	public MongoDBAtlasVectorStore mongoDBAtlasVectorStore(MongoTemplate client,
			MongoDbVectorStoreProperties properties, ObjectProvider<ObservationRegistry> observationRegistry,
			ObjectProvider<VectorStoreObservationConvention> customObservationConvention,
			BatchingStrategy batchingStrategy) {

		var builder = MongoDBAtlasVectorStore.builder(client, rawTextEmbeddingModel())
			.batchingStrategy(batchingStrategy)
			.observationRegistry(observationRegistry.getIfUnique(() -> ObservationRegistry.NOOP))
			.customObservationConvention(customObservationConvention.getIfAvailable(() -> null));
		if (properties.getDefaultTopK() >= 0) {
			builder.numCandidates(properties.getDefaultTopK());
		}

		if (properties.getDefaultSimilarityThreshold() >= 0.0) {
			builder.maxDistance(1.0f - properties.getDefaultSimilarityThreshold());
		}

		if (properties.getCollectName() != null) {
			builder.collectionName(properties.getCollectName());
		}

		if (properties.getNamespace() != null) {
			builder.pathName(properties.getNamespace());
		}
		return builder.build();
	}

}
