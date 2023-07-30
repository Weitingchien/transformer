import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pprint import pprint
from IPython.display import clear_output


import tensorflow as tf
import tensorflow_datasets as tfds

import logging

# logging.basicConfig(level="error")

# 不要用科學記號顯示較大的數字
np.set_printoptions(suppress=True)

print(tf.__version__)


def main():
    output_dir = "nmt"
    en_vocab_file = os.path.join(output_dir, r"en_vocab")
    zh_vocab_file = os.path.join(output_dir, r"zh_vocab")
    checkpoint_path = os.path.join(output_dir, r"checkpoints")
    log_dir = os.path.join(output_dir, r"logs")
    download_dir = r"tensorflow-datasets/downloads"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tmp_builder = tfds.builder("wmt19_translate/zh-en")
    pprint(tmp_builder.subsets)

    config = tfds.translate.wmt.WmtConfig(
        version=tfds.core.Version("0.0.3"),
        language_pair=("zh", "en"),
        # 定義子集，表示只下載newscommentary_v14
        subsets={tfds.Split.TRAIN: ["newscommentary_v14"]},
    )
    builder = tfds.builder("wmt_translate", config=config)
    # 下載和準備數據集，下載的路徑在download_dir目錄下
    builder.download_and_prepare(download_dir=download_dir)
    clear_output()

    # 獲取原始訓練數據集
    train_data = builder.as_dataset(split=tfds.Split.TRAIN, as_supervised=True)
    print(train_data)

    # 分割比例
    train_perc = 0.8
    val_perc = 0.1
    drop_perc = 0.1

    # 使用 tfds.Subset 來實現分割
    num_examples = len(train_data)
    train_size = int(num_examples * train_perc)
    print(f"train_size: {train_size}")
    val_size = int(num_examples * val_perc)
    drop_size = num_examples - train_size - val_size

    # 分割數據集
    train_data = train_data.take(train_size)
    print(f"train_data: {train_data}")
    val_data = train_data.skip(train_size).take(val_size)
    drop_data = train_data.skip(train_size + val_size)

    """
    for en, zh in train_data.take(3):
        print(en)
        print(zh)
        print("-" * 10)
    """

    sample_examples = []
    num_samples = 10

    for en_t, zh_t in train_data.take(num_samples):
        en = en_t.numpy().decode("utf-8")
        zh = zh_t.numpy().decode("utf-8")

        print(en)
        print(zh)
        print("-" * 10)

    # 之後用來簡單評估模型的訓練情況
    sample_examples.append((en, zh))

    # 建立字典來將每個單字轉成索引，使用SubwordTextEncoder掃整個資料集並建立字典
    try:
        subword_encoder_en = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            en_vocab_file
        )
        print(f"載入已建立的字典： {en_vocab_file}")
    except:
        print("沒有已建立的字典，從頭建立。")
        subword_encoder_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for en, _ in train_data), target_vocab_size=2**13
        )  # 有需要可以調整字典大小

        # 將字典檔案存下以方便下次 warmstart
        subword_encoder_en.save_to_file(en_vocab_file)

    print(f"字典大小：{subword_encoder_en.vocab_size}")
    print(f"前 10 個 subwords: {subword_encoder_en.subwords[:10]}")
    print()

    try:
        subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            zh_vocab_file
        )
        print(f"載入已建立的字典： {zh_vocab_file}")
    except:
        print("沒有已建立的字典，從頭建立。")
        subword_encoder_zh = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (zh.numpy() for _, zh in train_data),
            target_vocab_size=2**13,  # 有需要可以調整字典大小
            max_subword_length=1,
        )  # 每一個中文字就是字典裡的一個單位

    # 將字典檔案存下以方便下次 warmstart
    subword_encoder_zh.save_to_file(zh_vocab_file)

    print(f"字典大小：{subword_encoder_zh.vocab_size}")
    print(f"前 10 個 subwords: {subword_encoder_zh.subwords[:10]}")
    print()

    sample_string = sample_examples[0][1]
    indices = subword_encoder_zh.encode(sample_string)
    print(sample_string)
    print(indices)

    def encode(en_t, zh_t):
        # 因為字典的索引從 0 開始，
        # 我們可以使用 subword_encoder_en.vocab_size 這個值作為 BOS 的索引值
        # 用 subword_encoder_en.vocab_size + 1 作為 EOS 的索引值
        # 　subword_encoder_en.encode: 將單詞 "unbelievable" 拆分成 "un-", "believe", "able" 三個子詞。
        en_indices = (
            [subword_encoder_en.vocab_size]
            + subword_encoder_en.encode(en_t.numpy())
            + [subword_encoder_en.vocab_size + 1]
        )
        # 同理，不過是使用中文字典的最後一個索引 + 1
        zh_indices = (
            [subword_encoder_zh.vocab_size]
            + subword_encoder_zh.encode(zh_t.numpy())
            + [subword_encoder_zh.vocab_size + 1]
        )

        return en_indices, zh_indices

    en_t, zh_t = next(iter(train_data))
    en_indices, zh_indices = encode(en_t, zh_t)
    print("英文 BOS 的 index:", subword_encoder_en.vocab_size)
    print("英文 EOS 的 index:", subword_encoder_en.vocab_size + 1)
    print("中文 BOS 的 index:", subword_encoder_zh.vocab_size)
    print("中文 EOS 的 index:", subword_encoder_zh.vocab_size + 1)

    print("\n輸入為 2 個 Tensors:")
    pprint((en_t, zh_t))
    print("-" * 15)
    print("輸出為 2 個索引序列：")
    pprint((en_indices, zh_indices))

    def tf_encode(en_t, zh_t):
        # 在 `tf_encode` 函式裡頭的 `en_t` 與 `zh_t` 都不是 Eager Tensors
        # 要到 `tf.py_funtion` 裡頭才是
        # 另外因為索引都是整數，所以使用 `tf.int64`
        return tf.py_function(encode, [en_t, zh_t], [tf.int64, tf.int64])

    # `tmp_dataset` 為說明用資料集，說明完所有重要的 func，
    # 我們會從頭建立一個正式的 `train_dataset`
    tmp_dataset = train_data.map(tf_encode)
    en_indices, zh_indices = next(iter(tmp_dataset))
    print(en_indices)
    print(zh_indices)

    MAX_LENGTH = 40

    def filter_max_length(en, zh, max_length=MAX_LENGTH):
        # en, zh 分別代表英文與中文的索引序列
        return tf.logical_and(tf.size(en) <= max_length, tf.size(zh) <= max_length)

    # tf.data.Dataset.filter(func) 只會回傳 func 為真的例子
    tmp_dataset = tmp_dataset.filter(filter_max_length)
    # 因為我們數據量小可以這樣 count
    num_examples = 0
    for en_indices, zh_indices in tmp_dataset:
        cond1 = len(en_indices) <= MAX_LENGTH
        cond2 = len(zh_indices) <= MAX_LENGTH
        assert cond1 and cond2
        num_examples += 1

    print(f"所有英文與中文序列長度都不超過 {MAX_LENGTH} 個 tokens")
    print(f"訓練資料集裡總共有 {num_examples} 筆數據")

    BATCH_SIZE = 64
    # padded_batch: 將序列補0到batch裡最長序列的長度
    tmp_dataset = tmp_dataset.padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    en_batch, zh_batch = next(iter(tmp_dataset))
    print("英文索引序列的 batch")
    print(en_batch)
    print("-" * 20)
    print("中文索引序列的 batch")
    print(zh_batch)

    MAX_LENGTH = 40
    BATCH_SIZE = 128
    BUFFER_SIZE = 15000

    # 訓練集
    train_dataset = (
        train_data.map(tf_encode)  # 輸出：(英文句子, 中文句子)  # 輸出：(英文索引序列, 中文索引序列)
        .filter(filter_max_length)  # 同上，且序列長度都不超過 40
        .cache()  # 加快讀取數據
        .shuffle(BUFFER_SIZE)  # 將例子洗牌確保隨機性
        .padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))  # 將 batch 裡的序列都 pad 到一樣長度
        .prefetch(tf.data.experimental.AUTOTUNE)
    )  # 加速
    # 驗證集
    val_dataset = (
        val_data.map(tf_encode)
        .filter(filter_max_length)
        .padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
    )

    en_batch, zh_batch = next(iter(train_dataset))
    print("英文索引序列的 batch")
    print(en_batch)
    print("-" * 20)
    print("中文索引序列的 batch")
    print(zh_batch)

    # shape(batch_size, seq_len): batch_size代表句子的數量、seq_len代表句子的長度

    # 建立兩個要來持續追蹤的中英句子
    demo_examples = [
        ("It is important.", "这很重要。"),
        ("The numbers speak for themselves.", "数字证明了一切。"),
    ]
    # pprint(demo_examples)

    batch_size = 2
    demo_examples = tf.data.Dataset.from_tensor_slices(
        ([en for en, _ in demo_examples], [zh for _, zh in demo_examples])
    )

    # 將兩個句子透過之前定義的字典轉換成子詞的序列（sequence of subwords）
    # 並添加 padding token: <pad> 來確保 batch 裡的句子有一樣長度
    demo_dataset = demo_examples.map(tf_encode).padded_batch(
        batch_size, padded_shapes=([-1], [-1])
    )

    # 取出這個 demo dataset 裡唯一一個 batch
    inp, tar = next(iter(demo_dataset))
    print("inp:", inp)
    print("" * 10)
    print("tar:", tar)

    # 將索引序列丟入神經網路之前，還會作詞嵌入(word embedding)的處理

    # + 2 是因為我們額外加了 <start> 以及 <end> tokens
    vocab_size_en = subword_encoder_en.vocab_size + 2
    vocab_size_zh = subword_encoder_zh.vocab_size + 2

    # 為了方便 demo, 將詞彙轉換到一個 4 維的詞嵌入空間
    d_model = 4
    embedding_layer_en = tf.keras.layers.Embedding(vocab_size_en, d_model)
    embedding_layer_zh = tf.keras.layers.Embedding(vocab_size_zh, d_model)

    emb_inp = embedding_layer_en(inp)
    emb_tar = embedding_layer_zh(tar)

    def create_padding_mask(seq):
        # padding mask 的工作就是把索引序列中為 0 的位置設為 1
        mask = tf.cast(tf.equal(seq, 0), tf.float32)
        return mask[:, tf.newaxis, tf.newaxis, :]  # 　broadcasting

    inp_mask = create_padding_mask(inp)
    print(f"inp_mask: {inp_mask}")

    # tf.squeeze: 把多餘的維度去除，用來方便與inp比較
    print("inp:", inp)
    print("-" * 20)
    print("tf.squeeze(inp_mask):", tf.squeeze(inp_mask))

    # 設定一個 seed 確保我們每次都拿到一樣的隨機結果
    tf.random.set_seed(9527)
    # 自注意力機制：查詢 `q` 跟鍵值 `k` 都是 `emb_inp`
    q = emb_inp
    k = emb_inp
    # 簡單產生一個跟 `emb_inp` 同樣 shape 的 binary vector
    v = tf.cast(
        tf.math.greater(tf.random.uniform(shape=emb_inp.shape), 0.5), tf.float32
    )
    print(f"v: {v}")


if __name__ == "__main__":
    main()
