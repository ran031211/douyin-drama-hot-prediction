-- =====================================================================
--  抖音微短剧数据仓库 · 建库建表脚本
--  课题：基于 XGBoost 的抖音微短剧热度预测与影响因素分析
-- ---------------------------------------------------------------------
--  数据来源：抖音「AI 短剧榜」真实采集（2026-05 ~ 2026-10）
--  分析单元：一部微短剧（drama）
--  分层设计：
--      ODS 原始层  drama_info / drama_hot_comment / drama_ep1_comment
--      DWS 汇总层  drama_comment_agg（评论侧聚合宽表）
--      ADS 应用层  drama_feature_wide（建模特征宽表）
--      VW  视图层  v_heat_rank / v_genre_heat / v_comment_signal
--  字符集：utf8mb4（短剧名与评论含 emoji）
-- =====================================================================

CREATE DATABASE IF NOT EXISTS douyin_drama
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;
USE douyin_drama;

-- ---------------------------------------------------------------------
-- ODS-1  剧集主表：榜单剧集清单 + 热度 + 互动 + 简介
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS drama_info;
CREATE TABLE drama_info (
    drama_id          BIGINT       NOT NULL COMMENT '短剧唯一 ID（抖音作品 ID）',
    drama_name        VARCHAR(255) NOT NULL COMMENT '短剧名称',
    category          VARCHAR(255) NOT NULL COMMENT '题材分类（"/"分隔的多标签）',
    episodes          INT          NOT NULL COMMENT '总集数',
    heat_text         VARCHAR(64)           COMMENT '榜单热度原文，如 6849万热度',
    heat_value        BIGINT       NOT NULL COMMENT '热度数值（解析后）',
    like_count        BIGINT                COMMENT '点赞量',
    collect_count     BIGINT                COMMENT '收藏量',
    hot_rating_avg    DECIMAL(4,2)          COMMENT '热评均分（榜单口径，1~10）',
    rank_no           INT                   COMMENT '榜单排名',
    page_no           INT                   COMMENT '榜单页码',
    rank_tag          VARCHAR(64)           COMMENT '榜单标签，如 热播AI剧榜第1名',
    rank_tag_present  TINYINT      NOT NULL DEFAULT 0 COMMENT '是否带榜单标签 0/1',
    summary           TEXT                  COMMENT '剧情简介',
    created_at        TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (drama_id),
    KEY idx_heat (heat_value),
    KEY idx_rank (rank_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='微短剧剧集主表（ODS）';

-- ---------------------------------------------------------------------
-- ODS-2  热评事实表：每部剧的 Top 热评（含用户评分）
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS drama_hot_comment;
CREATE TABLE drama_hot_comment (
    comment_id    VARCHAR(64)  NOT NULL COMMENT '评论主键（剧集ID_序号）',
    drama_id      BIGINT       NOT NULL COMMENT '所属短剧 ID',
    drama_name    VARCHAR(255)          COMMENT '短剧名称（冗余，便于排查）',
    user_nick     VARCHAR(128)          COMMENT '评论人昵称',
    digg          INT          NOT NULL DEFAULT 0 COMMENT '评论点赞数',
    rating        TINYINT               COMMENT '用户评分 1~10',
    content       TEXT                  COMMENT '评论正文（已清洗）',
    content_len   INT                   COMMENT '评论字数',
    comment_time  DATETIME              COMMENT '评论时间',
    PRIMARY KEY (comment_id),
    KEY idx_drama (drama_id),
    CONSTRAINT fk_hot_drama FOREIGN KEY (drama_id)
        REFERENCES drama_info (drama_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='短剧热评事实表（ODS）';

-- ---------------------------------------------------------------------
-- ODS-3  第一集评论事实表：冷启动阶段最强的观众反馈信号
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS drama_ep1_comment;
CREATE TABLE drama_ep1_comment (
    comment_id    VARCHAR(64)  NOT NULL COMMENT '评论主键（剧集ID_序号）',
    drama_id      BIGINT       NOT NULL COMMENT '所属短剧 ID',
    drama_name    VARCHAR(255)          COMMENT '短剧名称（冗余）',
    user_nick     VARCHAR(128)          COMMENT '评论人昵称',
    digg          INT          NOT NULL DEFAULT 0 COMMENT '评论点赞数',
    content       TEXT                  COMMENT '评论正文（已清洗）',
    content_len   INT                   COMMENT '评论字数',
    comment_time  DATETIME              COMMENT '评论时间',
    PRIMARY KEY (comment_id),
    KEY idx_drama (drama_id),
    CONSTRAINT fk_ep1_drama FOREIGN KEY (drama_id)
        REFERENCES drama_info (drama_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='短剧第一集评论事实表（ODS）';

-- ---------------------------------------------------------------------
-- DWS  评论侧聚合宽表：剧集粒度的观众反馈指标
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS drama_comment_agg;
CREATE TABLE drama_comment_agg (
    drama_id          BIGINT NOT NULL COMMENT '短剧 ID',
    hot_cmt_cnt       INT    COMMENT '热评条数',
    hot_digg_mean     DECIMAL(10,2) COMMENT '热评平均点赞',
    hot_digg_max      INT    COMMENT '最高热评点赞',
    hot_rating_mean   DECIMAL(4,2) COMMENT '热评均分（明细口径）',
    hot_rating_std    DECIMAL(4,2) COMMENT '评分分歧度（标准差）',
    hot_rating_good   DECIMAL(5,4) COMMENT '好评率（≥9分占比）',
    hot_rating_bad    DECIMAL(5,4) COMMENT '差评率（≤3分占比）',
    ep1_cmt_cnt       INT    COMMENT '第一集评论数',
    ep1_digg_mean     DECIMAL(10,2) COMMENT '第一集评论平均点赞',
    ep1_digg_max      INT    COMMENT '第一集最高评论点赞',
    ep1_hot_ratio     DECIMAL(5,4) COMMENT '第一集百赞评论占比',
    ep1_super_ratio   DECIMAL(5,4) COMMENT '第一集千赞评论占比',
    ep1_urge_ratio    DECIMAL(5,4) COMMENT '催更评论占比',
    ep1_pos_ratio     DECIMAL(5,4) COMMENT '正面情感评论占比',
    ep1_neg_ratio     DECIMAL(5,4) COMMENT '负面情感评论占比',
    PRIMARY KEY (drama_id),
    CONSTRAINT fk_agg_drama FOREIGN KEY (drama_id)
        REFERENCES drama_info (drama_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评论侧聚合宽表（DWS）';

-- ---------------------------------------------------------------------
-- ADS  建模特征宽表：一类样本一行，直接供 XGBoost 训练消费
--      特征严格区分「上线前内容属性」与「上线后观众反馈」两组
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS drama_feature_wide;
CREATE TABLE drama_feature_wide (
    drama_id       BIGINT NOT NULL COMMENT '短剧 ID',
    drama_name     VARCHAR(255),
    category       VARCHAR(255),
    -- 标签
    heat_value     BIGINT       COMMENT '热度数值（原始目标）',
    log_heat       DECIMAL(10,6) COMMENT 'log10(热度)，回归目标',
    is_hot         TINYINT      COMMENT '是否爆款（热度 Top20%）',
    heat_tier      TINYINT      COMMENT '热度分档 0~3',
    heat_threshold BIGINT       COMMENT '爆款阈值',
    -- 内容属性（上线前可得）
    episodes        INT,
    log_episodes    DECIMAL(8,4),
    is_long_series  TINYINT,
    n_genre_tags    INT,
    title_len       INT,
    summary_len     INT,
    -- 观众口碑
    hot_rating_avg  DECIMAL(4,2),
    hot_rating_mean DECIMAL(4,2),
    hot_rating_std  DECIMAL(4,2),
    hot_rating_good DECIMAL(5,4),
    hot_rating_bad  DECIMAL(5,4),
    -- 评论互动
    hot_cmt_cnt     INT,
    hot_digg_mean   DECIMAL(10,2),
    log_hot_digg_sum DECIMAL(10,4),
    ep1_cmt_cnt     INT,
    ep1_digg_mean   DECIMAL(10,2),
    log_ep1_digg_sum DECIMAL(10,4),
    ep1_hot_ratio   DECIMAL(5,4),
    ep1_super_ratio DECIMAL(5,4),
    -- 文本态度
    ep1_text_len    DECIMAL(8,2),
    ep1_emoji_ratio DECIMAL(5,4),
    ep1_urge_ratio  DECIMAL(5,4),
    ep1_pos_ratio   DECIMAL(5,4),
    ep1_neg_ratio   DECIMAL(5,4),
    PRIMARY KEY (drama_id),
    KEY idx_hot (is_hot)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='建模特征宽表（ADS）';

-- =====================================================================
--  视图层：面向分析的预置查询
-- =====================================================================

-- V1 热度排行榜：附题材与互动概况
CREATE OR REPLACE VIEW v_heat_rank AS
SELECT d.rank_no,
       d.drama_name,
       d.category,
       d.episodes,
       d.heat_value,
       ROUND(d.heat_value / 10000, 0)             AS heat_wan,
       d.hot_rating_avg,
       a.hot_cmt_cnt,
       a.ep1_cmt_cnt,
       a.ep1_digg_mean,
       CASE WHEN d.rank_tag <> '' THEN d.rank_tag ELSE NULL END AS rank_tag
FROM drama_info d
LEFT JOIN drama_comment_agg a ON d.drama_id = a.drama_id
ORDER BY d.heat_value DESC;

-- V2 题材热度画像：按单标签展开，统计各题材的剧数与热度
CREATE OR REPLACE VIEW v_genre_heat AS
SELECT TRIM(SUBSTRING_INDEX(SUBSTRING_INDEX(d.category, '/', n.n), '/', -1)) AS genre,
       COUNT(*)                       AS drama_cnt,
       ROUND(AVG(d.heat_value)/10000, 0) AS avg_heat_wan,
       ROUND(MAX(d.heat_value)/10000, 0) AS max_heat_wan,
       SUM(f.is_hot)                  AS hot_cnt,
       ROUND(AVG(f.is_hot)*100, 2)    AS hot_rate_pct
FROM drama_info d
JOIN drama_feature_wide f ON d.drama_id = f.drama_id
JOIN (SELECT 1 n UNION SELECT 2 UNION SELECT 3 UNION SELECT 4
      UNION SELECT 5 UNION SELECT 6 UNION SELECT 7 UNION SELECT 8
      UNION SELECT 9 UNION SELECT 10) n
  ON n.n <= 1 + LENGTH(d.category) - LENGTH(REPLACE(d.category, '/', ''))
GROUP BY genre
HAVING drama_cnt >= 19
ORDER BY avg_heat_wan DESC;

-- V3 评论信号视图：把"口碑 / 互动 / 态度"三类信号并排展示
CREATE OR REPLACE VIEW v_comment_signal AS
SELECT d.drama_name,
       d.heat_value,
       f.is_hot,
       f.hot_rating_mean   AS 口碑评分,
       f.hot_rating_std    AS 评分分歧,
       f.ep1_digg_mean     AS 第一集点赞,
       f.ep1_hot_ratio     AS 百赞占比,
       f.ep1_urge_ratio    AS 催更率,
       f.ep1_pos_ratio     AS 正面率,
       f.ep1_neg_ratio     AS 负面率
FROM drama_info d
JOIN drama_feature_wide f ON d.drama_id = f.drama_id
ORDER BY d.heat_value DESC;
