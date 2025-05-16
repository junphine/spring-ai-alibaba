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

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import io.micrometer.observation.ObservationRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.springframework.ai.chat.metadata.DefaultUsage;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.document.Document;
import org.springframework.ai.document.MetadataMode;
import org.springframework.ai.embedding.*;
import org.springframework.ai.embedding.observation.DefaultEmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationContext;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationConvention;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationDocumentation;
import org.springframework.ai.model.ModelOptionsUtils;
import org.springframework.ai.retry.RetryUtils;
import org.springframework.lang.Nullable;
import org.springframework.retry.support.RetryTemplate;
import org.springframework.util.Assert;
import org.springframework.util.MimeTypeUtils;

/**
 * DashScope Embedding Model implementation.
 *
 * @author nuocheng.lxm
 * @author why_ohh
 * @author yuluo
 * @author <a href="mailto:550588941@qq.com">why_ohh</a>
 * @since 2024/7/31 10:57
 */
public class RawTextEmbeddingModel implements EmbeddingModel {

	private static final Logger logger = LoggerFactory.getLogger(RawTextEmbeddingModel.class);

	private static final EmbeddingModelObservationConvention DEFAULT_OBSERVATION_CONVENTION = new DefaultEmbeddingModelObservationConvention();

	public static final String DEFAULT_EMBEDDING_MODEL = "m3e-base";

	public static final String DEFAULT_EMBEDDING_TEXT_TYPE = "document";

	public static final String PROVIDER_NAME = "ignite";

	private final AtomicLong index = new AtomicLong(0);

	private final RawTextEmbeddingOptions defaultOptions;

	private final RetryTemplate retryTemplate;

	private final MetadataMode metadataMode;

	/**
	 * Observation registry used for instrumentation.
	 */
	private final ObservationRegistry observationRegistry;

	/**
	 * Conventions to use for generating observations.
	 */
	private EmbeddingModelObservationConvention observationConvention = DEFAULT_OBSERVATION_CONVENTION;

	public RawTextEmbeddingModel() {
		this(MetadataMode.EMBED);
	}

	public RawTextEmbeddingModel(MetadataMode metadataMode) {
		this(metadataMode,
				RawTextEmbeddingOptions.builder()
					.withModel(DEFAULT_EMBEDDING_MODEL)
					.withTextType(DEFAULT_EMBEDDING_TEXT_TYPE)
					.build(),
				RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	public RawTextEmbeddingModel(MetadataMode metadataMode, RawTextEmbeddingOptions dashScopeEmbeddingOptions) {
		this(metadataMode, dashScopeEmbeddingOptions, RetryUtils.DEFAULT_RETRY_TEMPLATE);
	}

	public RawTextEmbeddingModel(MetadataMode metadataMode, RawTextEmbeddingOptions dashScopeEmbeddingOptions,
			RetryTemplate retryTemplate) {
		this(metadataMode, dashScopeEmbeddingOptions, retryTemplate, ObservationRegistry.NOOP);
	}

	public RawTextEmbeddingModel(MetadataMode metadataMode, RawTextEmbeddingOptions options,
			RetryTemplate retryTemplate, ObservationRegistry observationRegistry) {

		Assert.notNull(metadataMode, "metadataMode must not be null");
		Assert.notNull(options, "options must not be null");
		Assert.notNull(retryTemplate, "retryTemplate must not be null");
		Assert.notNull(observationRegistry, "observationRegistry must not be null");

		this.metadataMode = metadataMode;
		this.defaultOptions = options;
		this.retryTemplate = retryTemplate;
		this.observationRegistry = observationRegistry;
	}

	@Override
	public float[] embed(Document document) {
		Assert.notNull(document, "Document must not be null");
		return this.embed(document.getFormattedContent(this.metadataMode));
	}

	public String formattedContent(Document document) {
		Assert.notNull(document, "Document must not be null");
		return document.getFormattedContent(this.metadataMode);
	}

	@Override
	public EmbeddingResponse call(EmbeddingRequest request) {
		var observationContext = EmbeddingModelObservationContext.builder()
			.embeddingRequest(request)
			.provider(PROVIDER_NAME)
			.build();

		return EmbeddingModelObservationDocumentation.EMBEDDING_MODEL_OPERATION
			.observation(this.observationConvention, DEFAULT_OBSERVATION_CONVENTION, () -> observationContext,
					this.observationRegistry)
			.observe(() -> {
				AtomicInteger atom = new AtomicInteger(0);
				List<Embedding> embeddings = request.getInstructions().stream().map(e -> {
					atom.getAndAdd(e.length());
					long docId = index.getAndAdd(1);
					EmbeddingResultMetadata metadata = new EmbeddingResultMetadata(String.valueOf(docId),
							EmbeddingResultMetadata.ModalityType.TEXT, MimeTypeUtils.TEXT_PLAIN, e);
					return new Embedding(null, 0, metadata);
				}).toList();

				DefaultUsage usage = new DefaultUsage(atom.intValue(), 0, atom.intValue());

				var metadata = generateResponseMetadata(request.getOptions().getModel(), usage);
				EmbeddingResponse embeddingResponse = new EmbeddingResponse(embeddings, metadata);
				observationContext.setResponse(embeddingResponse);

				return embeddingResponse;
			});
	}

	/**
	 * Merge runtime and default {@link EmbeddingOptions} to compute the final options to
	 * use in the request.
	 */
	private RawTextEmbeddingOptions mergeOptions(@Nullable EmbeddingOptions runtimeOptions,
			RawTextEmbeddingOptions defaultOptions) {
		if (runtimeOptions == null) {
			return defaultOptions;
		}

		return RawTextEmbeddingOptions.builder()
			// Handle portable embedding options
			.withModel(ModelOptionsUtils.mergeOption(runtimeOptions.getModel(), defaultOptions.getModel()))
			.withDimensions(
					ModelOptionsUtils.mergeOption(runtimeOptions.getDimensions(), defaultOptions.getDimensions()))
			// Handle DashScope specific embedding options
			.withTextType(defaultOptions.getTextType())
			.build();
	}

	private EmbeddingResponseMetadata generateResponseMetadata(String model, Usage usage) {
		Map<String, Object> map = new HashMap<>();
		map.put("model", model);
		map.put("total-tokens", usage.getTotalTokens());

		return new EmbeddingResponseMetadata(model, usage, map);
	}

	/**
	 * Use the provided convention for reporting observation data
	 * @param observationConvention The provided convention
	 */
	public void setObservationConvention(EmbeddingModelObservationConvention observationConvention) {
		Assert.notNull(observationConvention, "observationConvention cannot be null");
		this.observationConvention = observationConvention;
	}

	@Override
	public int dimensions() {
		return defaultOptions.getDimensions() == null ? 768 : defaultOptions.getDimensions();
	}

	public RawTextEmbeddingOptions options() {
		return defaultOptions;
	}

}
