#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JGB利回り曲線の主成分分析（PCA）を行うスクリプト

使い方の例:
  python pca_yieldcurve.py
  python pca_yieldcurve.py --csv jgbcm_all.csv --encoding cp932 --n_components 3 --save pca.png

このスクリプトは以下の手順で処理します：
 1. CSV を読み込み（デフォルト: jgbcm_all.csv）
 2. 最初の行をヘッダとして使い、"基準日" 列をインデックスにする
 3. 欠損値や "-" を除去して数値データに変換
 4. sklearn.decomposition.PCA で主成分を抽出
 5. 主成分スコアをプロットし、必要なら画像ファイルに保存
"""
from pathlib import Path
from typing import Optional, List

import argparse
import sys

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def load_jgb_csv(path: Path, encoding: str = "cp932") -> pd.DataFrame:
    """
    CSVを読み込み、最初の行をヘッダとして使って日付インデックス付きの数値データを返す。
    ファイル内の "-" を欠損値として扱い、すべての列を数値型に変換した上で
    欠損値がある行は除去します。
    """
    df = pd.read_csv(path, encoding=encoding, dtype=str)
    if df.shape[0] < 2:
        raise ValueError("CSV にヘッダ行とデータ行が必要です。")
    header = df.iloc[0]
    data = df.iloc[1:].copy()
    data.columns = header

    # 日本語の列名 "基準日" を "date" にする（存在する場合）
    if "基準日" in data.columns:
        data = data.rename(columns={"基準日": "date"})
    if "date" not in data.columns:
        raise KeyError("CSV に '基準日'（または 'date'）列が見つかりません。")

    data.set_index("date", inplace=True)

    # "-" を欠損値扱いにして数値型へ
    data = data.replace("-", pd.NA)

    # pandas の nullable float 型を使って変換し、欠損値がある行は除去
    num_df = data.astype("Float64").dropna(how="any")

    if num_df.empty:
        raise ValueError("数値データに変換した結果、利用可能な行がありません。")

    return num_df


def compute_pca(df: pd.DataFrame, n_components: int = 3) -> (pd.DataFrame, PCA):
    """
    PCA を実行し、主成分スコア（サンプル x コンポーネント）を DataFrame として返す。
    """
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(df.to_numpy())

    # コンポーネント名の決定（n_components == 3 のときは Level/Slope/Curvature を使う）
    if n_components == 3:
        columns: List[str] = ["Level", "Slope", "Curvature"]
    else:
        columns = [f"PC{i+1}" for i in range(n_components)]

    pc_df = pd.DataFrame(components, columns=columns, index=df.index)
    return pc_df, pca


def plot_pcs(pc_df: pd.DataFrame, title: str = "JGB Yield Curve PCA", save_path: Optional[Path] = None) -> None:
    """
    主成分スコアをプロット。save_path が指定されていれば画像として保存、そうでなければ表示する。
    """
    ax = pc_df.plot(figsize=(12, 6))
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"プロットを保存しました: {save_path}")
    else:
        plt.show()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="JGB Yield Curve PCA")
    p.add_argument("--csv", default="jgbcm_all.csv", help="読み込むCSVファイル（デフォルト: jgbcm_all.csv）")
    p.add_argument("--encoding", default="cp932", help="CSVの文字エンコーディング（デフォルト: cp932）")
    p.add_argument("--n_components", type=int, default=3, help="PCA の主成分数（デフォルト: 3）")
    p.add_argument("--save", default=None, help="プロットを保存するパス（指定しない場合は表示する）")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"エラー: ファイルが見つかりません: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        num_df = load_jgb_csv(csv_path, encoding=args.encoding)
    except Exception as e:
        print(f"データ読み込みエラー: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        pc_df, pca = compute_pca(num_df, n_components=args.n_components)
    except Exception as e:
        print(f"PCA 実行中にエラーが発生しました: {e}", file=sys.stderr)
        sys.exit(1)

    # 分散説明率を表示
    var_ratio = pca.explained_variance_ratio_
    for i, vr in enumerate(var_ratio, start=1):
        print(f"PC{i} explained variance ratio: {vr:.4f}")
    print(f"Total explained variance (first {len(var_ratio)}): {var_ratio.sum():.4f}")

    save_path = Path(args.save) if args.save else None
    plot_pcs(pc_df, save_path=save_path)


if __name__ == "__main__":
    main()
