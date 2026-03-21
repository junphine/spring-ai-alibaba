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
import com.alibaba.fastjson.JSONObject;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.redisson.api.RBlockingQueue;
import org.redisson.api.RDelayedQueue;
import org.redisson.api.RedissonClient;
import org.springframework.stereotype.Component;
import org.springframework.util.CollectionUtils;

import java.util.Map;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * MQ Consumer Manager for Redis-based message queues.
 * Manages multiple consumers and provides message consumption capabilities using Redisson.
 *
 * @since 1.0.0.3
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class RedissonConsumerManager {

    /** Redisson client for Redis operations */
    private final RedissonClient redissonClient;

    /** MQ configuration properties */
    private final MqConfigProperties mqConfigProperties;

    /** Map of consumer groups to their corresponding consumer threads */
    private final Map<String, ConsumerWorker> consumerWorkerMap = new ConcurrentHashMap<>();

    /** Map of consumer groups to their scheduled executors for delayed queue cleanup */
    private final Map<String, ScheduledExecutorService> scheduledExecutorMap = new ConcurrentHashMap<>();

    /** Thread pool for consumer workers */
    private final ExecutorService consumerExecutor = Executors.newCachedThreadPool();

    /** 队列名称前缀 */
    private static final String QUEUE_PREFIX = "mq:queue:";

    /** 延迟队列名称前缀 */
    private static final String DELAY_QUEUE_PREFIX = "mq:delay:queue:";

    /** 消费者组前缀 */
    private static final String CONSUMER_GROUP_PREFIX = "mq:consumer:group:";

    /**
     * Subscribe to a topic with specified consumer group
     * @param group consumer group name
     * @param topic topic to subscribe (used as queue name)
     * @param handler message handler for processing messages
     */
    public void subscribe(String group, String topic, MqConsumerHandler<MqMessage> handler) {
        String queueName = QUEUE_PREFIX + topic;
        String consumerKey = CONSUMER_GROUP_PREFIX + group + ":" + topic;

        // Check if consumer already exists
        if (consumerWorkerMap.containsKey(consumerKey)) {
            log.warn("Consumer already exists for group: {}, topic: {}", group, topic);
            return;
        }

        try {
            RBlockingQueue<String> blockingQueue = redissonClient.getBlockingQueue(queueName);

            ConsumerWorker worker = new ConsumerWorker(
                    consumerKey,
                    blockingQueue,
                    handler,
                    group,
                    topic
            );

            consumerWorkerMap.put(consumerKey, worker);

            // Start consumer worker in thread pool
            consumerExecutor.submit(worker);

            // Start delayed queue cleanup scheduler if needed
            startDelayedQueueCleanup(group, topic);

            log.info("Subscribed to group: {}, topic: {}, queue: {}", group, topic, queueName);
        }
        catch (Exception e) {
            log.error("Failed to subscribe to group: {}, topic: {}", group, topic, e);
            throw new RuntimeException("Failed to subscribe", e);
        }
    }

    /**
     * Subscribe to a topic with specified consumer group and custom filter expression
     * @param group consumer group name
     * @param topic topic to subscribe (used as queue name)
     * @param filterExpression filter expression (e.g., tag pattern)
     * @param handler message handler for processing messages
     */
    public void subscribe(String group, String topic, String filterExpression, MqConsumerHandler<MqMessage> handler) {
        // For Redis implementation, filterExpression can be used for tag filtering
        // This is a simplified implementation - you may want to implement more sophisticated filtering
        if (filterExpression != null && !"*".equals(filterExpression)) {
            // Store filter expression for this consumer
            String filterKey = CONSUMER_GROUP_PREFIX + group + ":" + topic + ":filter";
            redissonClient.getBucket(filterKey).set(filterExpression);
            log.info("Set filter expression for group: {}, topic: {}, filter: {}", group, topic, filterExpression);
        }

        subscribe(group, topic, handler);
    }

    /**
     * Unsubscribe from a topic
     * @param group consumer group name
     * @param topic topic to unsubscribe
     */
    public void unsubscribe(String group, String topic) {
        String consumerKey = CONSUMER_GROUP_PREFIX + group + ":" + topic;

        ConsumerWorker worker = consumerWorkerMap.remove(consumerKey);
        if (worker != null) {
            worker.stop();
            log.info("Unsubscribed from group: {}, topic: {}", group, topic);
        }

        // Shutdown scheduled executor for this consumer
        ScheduledExecutorService scheduledExecutor = scheduledExecutorMap.remove(consumerKey);
        if (scheduledExecutor != null) {
            scheduledExecutor.shutdown();
            try {
                if (!scheduledExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                    scheduledExecutor.shutdownNow();
                }
            }
            catch (InterruptedException e) {
                scheduledExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Start delayed queue cleanup scheduler
     */
    private void startDelayedQueueCleanup(String group, String topic) {
        String consumerKey = CONSUMER_GROUP_PREFIX + group + ":" + topic;
        String delayQueueName = DELAY_QUEUE_PREFIX + topic;

        // Create scheduled executor for this consumer
        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduledExecutorMap.put(consumerKey, scheduler);

        // Schedule periodic cleanup of delayed queue (optional, as Redisson handles it automatically)
        scheduler.scheduleAtFixedRate(() -> {
            try {
                RDelayedQueue<String> delayedQueue = redissonClient.getDelayedQueue(
                        redissonClient.getBlockingQueue(delayQueueName)
                );
                // Redisson's delayed queue handles expiration automatically
                // This is just for monitoring purposes
                log.debug("Delayed queue cleanup check for topic: {}", topic);
            }
            catch (Exception e) {
                log.error("Error in delayed queue cleanup for topic: {}", topic, e);
            }
        }, 1, 10, TimeUnit.MINUTES);
    }

    /**
     * Get consumer status
     * @param group consumer group name
     * @param topic topic name
     * @return consumer status
     */
    public ConsumerStatus getConsumerStatus(String group, String topic) {
        String consumerKey = CONSUMER_GROUP_PREFIX + group + ":" + topic;
        ConsumerWorker worker = consumerWorkerMap.get(consumerKey);

        if (worker == null) {
            return ConsumerStatus.builder()
                    .active(false)
                    .messageCount(0)
                    .build();
        }

        return ConsumerStatus.builder()
                .active(worker.isRunning())
                .messageCount(worker.getMessageCount())
                .processedCount(worker.getProcessedCount())
                .failedCount(worker.getFailedCount())
                .build();
    }

    /**
     * Get queue size for a topic
     * @param topic topic name
     * @return queue size
     */
    public int getQueueSize(String topic) {
        String queueName = QUEUE_PREFIX + topic;
        RBlockingQueue<String> queue = redissonClient.getBlockingQueue(queueName);
        return queue.size();
    }

    /**
     * Clear queue for a topic
     * @param topic topic name
     */
    public void clearQueue(String topic) {
        String queueName = QUEUE_PREFIX + topic;
        RBlockingQueue<String> queue = redissonClient.getBlockingQueue(queueName);
        queue.clear();
        log.info("Cleared queue for topic: {}", topic);
    }

    /**
     * Shutdown all consumers gracefully
     */
    @PreDestroy
    public void shutdown() {
        log.info("Shutting down all consumers...");

        // Stop all consumer workers
        consumerWorkerMap.forEach((key, worker) -> {
            try {
                worker.stop();
                log.info("Consumer stopped: {}", key);
            }
            catch (Exception e) {
                log.error("Failed to stop consumer: {}", key, e);
            }
        });

        // Clear worker map
        consumerWorkerMap.clear();

        // Shutdown all scheduled executors
        scheduledExecutorMap.forEach((key, executor) -> {
            try {
                executor.shutdown();
                if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            }
            catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        });

        scheduledExecutorMap.clear();

        // Shutdown consumer thread pool
        consumerExecutor.shutdown();
        try {
            if (!consumerExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                consumerExecutor.shutdownNow();
            }
        }
        catch (InterruptedException e) {
            consumerExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        log.info("All consumers shutdown successfully");
    }

    /**
     * Consumer worker thread for processing messages
     */
    private class ConsumerWorker implements Runnable {
        private final String consumerKey;
        private final RBlockingQueue<String> queue;
        private final MqConsumerHandler<MqMessage> handler;
        private final String group;
        private final String topic;
        private final AtomicBoolean running = new AtomicBoolean(true);
        private final AtomicLong messageCount = new AtomicLong(0);
        private final AtomicLong processedCount = new AtomicLong(0);
        private final AtomicLong failedCount = new AtomicLong(0);

        public ConsumerWorker(String consumerKey, RBlockingQueue<String> queue,
                              MqConsumerHandler<MqMessage> handler, String group, String topic) {
            this.consumerKey = consumerKey;
            this.queue = queue;
            this.handler = handler;
            this.group = group;
            this.topic = topic;
        }

        @Override
        public void run() {
            while (running.get()) {
                try {
                    // Poll message with timeout
                    String messageJson = queue.poll(mqConfigProperties.getPollTimeoutMs(), TimeUnit.MILLISECONDS);

                    if (messageJson != null) {
                        messageCount.incrementAndGet();
                        processMessage(messageJson);
                    }
                }
                catch (InterruptedException e) {
                    if (running.get()) {
                        log.warn("Consumer worker interrupted for group: {}, topic: {}", group, topic);
                    }
                    Thread.currentThread().interrupt();
                    break;
                }
                catch (Exception e) {
                    log.error("Error in consumer worker for group: {}, topic: {}", group, topic, e);
                }
            }
            log.info("Consumer worker stopped for group: {}, topic: {}", group, topic);
        }

        /**
         * Process a single message
         */
        private void processMessage(String messageJson) {
            try {
                MqMessage message = buildMqMessageFromJson(messageJson);

                // Check filter if applicable
                String filterKey = CONSUMER_GROUP_PREFIX + group + ":" + topic + ":filter";
                String filterExpression = redissonClient.<String>getBucket(filterKey).get();

                if (filterExpression != null && !shouldProcess(message, filterExpression)) {
                    log.debug("Message filtered out by filter expression: {}", filterExpression);
                    processedCount.incrementAndGet();
                    return;
                }

                // Process message
                if (handler != null) {
                    handler.handle(message);
                    processedCount.incrementAndGet();
                }
                else {
                    log.warn("No handler provided for message, group: {}, topic: {}", group, topic);
                }
            }
            catch (Exception e) {
                failedCount.incrementAndGet();
                log.error("Failed to process message for group: {}, topic: {}", group, topic, e);

                // Handle failed message - could implement retry logic here
                handleFailedMessage(messageJson, e);
            }
        }

        /**
         * Check if message should be processed based on filter expression
         */
        private boolean shouldProcess(MqMessage message, String filterExpression) {
            // Simple tag-based filtering
            if (filterExpression == null || "*".equals(filterExpression)) {
                return true;
            }

            if (filterExpression.startsWith("tags=")) {
                String expectedTag = filterExpression.substring(5);
                return expectedTag.equals(message.getTag());
            }

            // Default to true for unsupported filter expressions
            return true;
        }

        /**
         * Handle failed message (implement retry or dead letter queue)
         */
        private void handleFailedMessage(String messageJson, Exception e) {
            try {
                // Send to dead letter queue
                String deadLetterQueueName = QUEUE_PREFIX + "dlq:" + topic;
                RBlockingQueue<String> dlq = redissonClient.getBlockingQueue(deadLetterQueueName);

                // Add retry count to message
                JSONObject jsonObject = JSON.parseObject(messageJson);
                int retryCount = jsonObject.getInteger("retryCount") == null ? 0 : jsonObject.getInteger("retryCount");
                jsonObject.put("retryCount", retryCount + 1);
                jsonObject.put("lastError", e.getMessage());
                jsonObject.put("lastErrorTime", System.currentTimeMillis());

                if (retryCount < mqConfigProperties.getMaxAttempts()) {
                    // Re-queue for retry
                    String retryQueueName = QUEUE_PREFIX + topic + ":retry";
                    RBlockingQueue<String> retryQueue = redissonClient.getBlockingQueue(retryQueueName);
                    retryQueue.add(jsonObject.toJSONString());
                    log.info("Message requeued for retry, retry count: {}, topic: {}", retryCount + 1, topic);
                }
                else {
                    // Send to dead letter queue
                    dlq.add(jsonObject.toJSONString());
                    log.warn("Message sent to dead letter queue after {} retries, topic: {}", retryCount, topic);
                }
            }
            catch (Exception ex) {
                log.error("Failed to handle failed message", ex);
            }
        }

        /**
         * Stop the consumer worker
         */
        public void stop() {
            running.set(false);
        }

        /**
         * Check if consumer is running
         */
        public boolean isRunning() {
            return running.get();
        }

        /**
         * Get total message count
         */
        public long getMessageCount() {
            return messageCount.get();
        }

        /**
         * Get processed message count
         */
        public long getProcessedCount() {
            return processedCount.get();
        }

        /**
         * Get failed message count
         */
        public long getFailedCount() {
            return failedCount.get();
        }
    }

    /**
     * Build MqMessage from JSON string
     */
    private MqMessage buildMqMessageFromJson(String messageJson) {
        return JSON.parseObject(messageJson, MqMessage.class);
    }

    /**
     * Consumer status DTO
     */
    @lombok.Data
    @lombok.Builder
    public static class ConsumerStatus {
        private boolean active;
        private long messageCount;
        private long processedCount;
        private long failedCount;
    }
}