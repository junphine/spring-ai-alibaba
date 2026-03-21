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

package com.alibaba.cloud.ai.studio.core.config;

import com.baomidou.mybatisplus.annotation.DbType;
import com.baomidou.mybatisplus.autoconfigure.ConfigurationCustomizer;
import com.baomidou.mybatisplus.core.MybatisConfiguration;
import com.baomidou.mybatisplus.extension.plugins.MybatisPlusInterceptor;
import com.baomidou.mybatisplus.extension.plugins.inner.PaginationInnerInterceptor;
import org.apache.ibatis.session.SqlSessionFactory;
import org.apache.ibatis.type.BooleanTypeHandler;
import org.apache.ibatis.type.JdbcType;
import org.apache.ibatis.type.TypeHandlerRegistry;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Configuration class for MyBatis-Plus integration. Provides pagination support and
 * mapper scanning configuration.
 *
 * @since 1.0.0.3
 */
@Configuration
@MapperScan("com.alibaba.cloud.ai.studio.core.base.mapper")
public class MybatisPlusConfig {

	/**
	 * Configures MyBatis-Plus interceptor with MySQL pagination support.
	 * @return MybatisPlusInterceptor instance
	 */
	@Bean
	public MybatisPlusInterceptor mybatisPlusInterceptor() {
		MybatisPlusInterceptor interceptor = new MybatisPlusInterceptor();
		interceptor.addInnerInterceptor(new PaginationInnerInterceptor(DbType.MYSQL));
		return interceptor;
	}

	@Bean
	public ConfigurationCustomizer mybatisConfigurationCustomizer() {
		return new ConfigurationCustomizer() {
			@Override
			public void customize(MybatisConfiguration configuration) {
				TypeHandlerRegistry typeHandlerRegistry = configuration.getTypeHandlerRegistry();
				// 全局注册 Boolean 类型处理器
				typeHandlerRegistry.register(new BooleanToSmallIntHandler());
				typeHandlerRegistry.register(Boolean.class, null, new BooleanToSmallIntHandler());
				typeHandlerRegistry.register(boolean.class, null, new BooleanToSmallIntHandler());
				typeHandlerRegistry.register(boolean.class, JdbcType.SMALLINT, new BooleanToSmallIntHandler());
				typeHandlerRegistry.register(boolean.class, JdbcType.BOOLEAN, new BooleanTypeHandler());
				typeHandlerRegistry.register(Boolean.class, JdbcType.BOOLEAN, new BooleanTypeHandler());
			}
		};
	}
}
