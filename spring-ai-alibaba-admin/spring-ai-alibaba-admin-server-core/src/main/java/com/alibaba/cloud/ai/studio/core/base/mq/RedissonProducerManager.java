/*
 * Copyright 2025 the original author or authors.
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

package com.alibaba.cloud.ai.studio.core.base.mq;

import com.alibaba.cloud.ai.studio.core.config.MqConfigProperties;
import com.alibaba.fastjson.JSON;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBlockingQueue;
import org.redisson.api.RDelayedQueue;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

/**
 * Manages Redis-based message producers and provides message sending capabilities.
 * Uses Redisson for distributed queue implementation.
 *
 * @since 1.0.0.3
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RedissonProducerManager {

    /** Redisson client for Redis operations */
    private final RedissonClient redissonClient;

    /** MQ configuration properties */
    private final MqConfigProperties mqConfigProperties;

    /** Thread pool for async operations */
    private final ExecutorService asyncExecutor = Executors.newCachedThreadPool();

    /**
     * 队列名称前缀
     */
    private static final String QUEUE_PREFIX = "mq:queue:";

    /**
     * 延迟队列名称前缀
     */
    private static final String DELAY_QUEUE_PREFIX = "mq:delay:queue:";

    /**
     * Sends a message synchronously
     * @param producer Producer identifier (can used as queue name,when one producer multi cosumners)
     * @param message Message to be sent
     * @return Send result containing message ID
     */
    public SendResult send(String producer, MqMessage message) {
        try {
            String queueName = QUEUE_PREFIX + message.getTopic();
            RBlockingQueue<String> queue = redissonClient.getBlockingQueue(queueName);

            String messageJson = JSON.toJSONString(message);
            queue.add(messageJson);

            log.debug("Message sent successfully to queue: {}, messageId: {}", queueName, message.getMessageId());
            return SendResult.builder().messageId(message.getMessageId()).build();
        }
        catch (Exception e) {
            log.error("Failed to send message, topic: {}", message.getTopic(), e);
            throw new RuntimeException("Failed to send message", e);
        }
    }

    /**
     * Sends a delayed message synchronously
     * @param producer Producer identifier
     * @param message Message to be sent
     * @param delaySeconds Delay time in seconds
     * @return Send result containing message ID
     */
    public SendResult sendDelay(String producer, MqMessage message, int delaySeconds) {
        try {

            String queueName = DELAY_QUEUE_PREFIX + message.getTopic();

            RBlockingQueue<String> blockingQueue = redissonClient.getBlockingQueue(queueName);
            RDelayedQueue<String> delayedQueue = redissonClient.getDelayedQueue(blockingQueue);

            String messageJson = JSON.toJSONString(message);
            delayedQueue.offer(messageJson, delaySeconds, TimeUnit.SECONDS);

            log.debug("Delayed message sent successfully to queue: {}, delay: {}s, messageId: {}",
                    queueName, delaySeconds, message.getMessageId());
            return SendResult.builder().messageId(message.getMessageId()).build();
        }
        catch (Exception e) {
            log.error("Failed to send delayed message, topic: {}, delay: {}s", message.getTopic(), delaySeconds, e);
            throw new RuntimeException("Failed to send delayed message", e);
        }
    }

    /**
     * Sends messages asynchronously with callback
     * @param producer Producer identifier (used as queue name)
     * @param messages List of messages to be sent
     * @param callback Callback to handle send results
     */
    public void sendAsync(String producer, List<MqMessage> messages, SendCallback callback) {
        try {
            List<CompletableFuture<Void>> futures = new ArrayList<>();

            for (MqMessage message : messages) {
                CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                    try {
                        send(producer, message);
                        callback.onSuccess(SendResult.builder().messageId(message.getMessageId()).build());
                    }
                    catch (Exception e) {
                        callback.onError(e);
                    }
                }, asyncExecutor);
                futures.add(future);
            }

            // Wait for all messages to be sent
            CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                    .get(mqConfigProperties.getSendMessageTimeoutMs(), TimeUnit.MILLISECONDS);
        }
        catch (Exception e) {
            log.error("Failed to async send message", e);
            throw new RuntimeException("Failed to async send message", e);
        }
    }

    /**
     * Sends messages asynchronously with success and error handlers
     * @param producer Producer identifier (used as queue name)
     * @param messages List of messages to be sent
     * @param onSuccess Success handler
     * @param onError Error handler
     */
    public void sendAsync(String producer, List<MqMessage> messages, Consumer<SendResult> onSuccess,
                          Consumer<Throwable> onError) {
        sendAsync(producer, messages, new SendCallback() {
            @Override
            public void onSuccess(SendResult sendResult) {
                if (onSuccess != null) {
                    onSuccess.accept(sendResult);
                }
            }

            @Override
            public void onError(Throwable e) {
                if (onError != null) {
                    onError.accept(e);
                }
            }
        });
    }

    /**
     * Sends a single message asynchronously
     * @param producer Producer identifier (used as queue name)
     * @param message Message to be sent
     * @return CompletableFuture with send result
     */
    public CompletableFuture<SendResult> sendAsync(String producer, MqMessage message) {
        return CompletableFuture.supplyAsync(() -> send(producer, message), asyncExecutor);
    }

    /**
     * Sends a delayed message asynchronously
     * @param producer Producer identifier (used as queue name)
     * @param message Message to be sent
     * @param delaySeconds Delay time in seconds
     * @return CompletableFuture with send result
     */
    public CompletableFuture<SendResult> sendDelayAsync(String producer, MqMessage message, int delaySeconds) {
        return CompletableFuture.supplyAsync(() -> sendDelay(producer, message, delaySeconds), asyncExecutor);
    }

    /**
     * Sends messages in batch synchronously
     * @param producer Producer identifier (used as queue name)
     * @param messages List of messages to be sent
     * @return List of send results
     */
    public List<SendResult> sendBatch(String producer, List<MqMessage> messages) {
        List<SendResult> results = new ArrayList<>();
        for (MqMessage message : messages) {
            results.add(send(producer, message));
        }
        return results;
    }

    /**
     * Sends messages in batch asynchronously
     * @param producer Producer identifier (used as queue name)
     * @param messages List of messages to be sent
     * @return CompletableFuture with list of send results
     */
    public CompletableFuture<List<SendResult>> sendBatchAsync(String producer, List<MqMessage> messages) {
        List<CompletableFuture<SendResult>> futures = new ArrayList<>();
        for (MqMessage message : messages) {
            futures.add(sendAsync(producer, message));
        }

        return CompletableFuture.allOf(futures.toArray(new CompletableFuture[0]))
                .thenApply(v -> {
                    List<SendResult> results = new ArrayList<>();
                    for (CompletableFuture<SendResult> future : futures) {
                        results.add(future.join());
                    }
                    return results;
                });
    }

    /**
     * 构建消息键
     */
    private String buildMessageKey(MqMessage message) {
        if (!CollectionUtils.isEmpty(message.getKeys())) {
            return String.join(":", message.getKeys());
        }
        return message.getMessageId();
    }
}

