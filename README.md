# ComfyUI-MiniMaxH3-SingleFrame

[日本語](#日本語) | [English](#english)

## 日本語

MiniMax H3で、短い互換フレーム列を生成して、その中から1枚を取り出すための実験的なComfyUIカスタムノードです。

通常の動画生成ノードとしてではなく、画像編集や始点/終点フレーム補間を静止画ワークフローに近い形で扱うことを目的にしています。

### ノード

- `MiniMax H3 Single Frame Edit`: 入力画像とプロンプトからMiniMax H3用のconditioningとAV latentを作ります。`frame_count` のデフォルトは `5` です。`keyframe` は入力画像を0番フレームに固定し、`reference` は `<Picture 1>` の参照画像として使います。
- `MiniMax H3 Start End Frame Interpolate`: `start_frame` を0番フレーム、`end_frame` を `frame_count - 1` に固定します。`frame_count` のデフォルトは `5` です。
- `MiniMax H3 Temporal RoPE Patch`: target video tokenの時間RoPEを指定フレームへ寄せる実験用MODELパッチです。静止画寄りの出力を狙う場合に使います。
- `Empty MiniMax H3 Single Frame Latent`: カスタムワークフロー向けにMiniMax H3互換の空AV latentを作ります。
- `MiniMax H3 Select Frame`: デコード後の画像列から1枚を選択します。デフォルトは0番フレームです。

### フレーム数

MiniMax H3互換の都合で、指定した `frame_count` は次の条件を満たす値に切り上げます。

```text
frame_count % 17 == 5
```

例:

```text
5  -> 5
6  -> 22
21 -> 22
22 -> 22
23 -> 39
```

最小値は `5` です。2フレームだけを直接生成するのではなく、5フレーム生成して必要なフレームを `MiniMax H3 Select Frame` で取り出します。

### リサイズ

入力画像はノード内部で指定サイズにストレッチされます。`resize mode` のUIはありません。

### Temporal RoPE Patch

`MiniMax H3 Temporal RoPE Patch` は、MODELを受け取ってMODELを返します。UNET Loaderの後、KSamplerへ入る前に接続してください。

- `frame_index`: 時間RoPEの基準にするlatentフレームです。デフォルトは `0` です。
- `strength`: 通常の時間RoPEから固定RoPEへ寄せる強さです。デフォルトは `1.0` で、指定フレームへ完全固定します。`0.5` で半分だけ寄せ、`0.0` でRoPE導入前と同じ挙動です。

デフォルトの `frame_index = 0`, `strength = 1.0` は静止画寄りの出力を狙う設定です。動き、色変化、時間方向のドリフトが減る可能性があります。効きが強すぎて崩れる場合は `strength = 0.25` から `0.75` の範囲で下げてください。

これは実験機能です。学習時の時間配置から外れるため、にじみ、反復模様、質感劣化、構図崩れが出る場合もあります。効果がない、または悪化する場合は `strength = 0.0` に戻してください。

### サンプルワークフロー

- `example_workflows/minimax_h3_single_frame_edit_basic.json`
- `example_workflows/minimax_h3_start_end_interpolate_middle.json`

ComfyUIでワークフローを開き、UNET、CLIP、VAE、入力画像のファイル名を手元のMiniMax H3環境に合わせて変更してください。
`sample` ディレクトリにはワークフロー確認用の入力画像例があります。

このカスタムノードはMiniMax H3本体のモデルコードを直接パッチしません。

## English

Experimental ComfyUI custom nodes for MiniMax H3 that generate a short compatible frame sequence and extract a single frame from it.

These nodes are intended for image-editing-like workflows and start/end frame interpolation, not as a general video generation wrapper.

### Nodes

- `MiniMax H3 Single Frame Edit`: Creates MiniMax H3 conditioning and an AV latent from an input image and prompt. `frame_count` defaults to `5`. `keyframe` anchors the input image at frame 0, while `reference` uses it as `<Picture 1>` reference conditioning.
- `MiniMax H3 Start End Frame Interpolate`: Anchors `start_frame` at frame 0 and `end_frame` at `frame_count - 1`. `frame_count` defaults to `5`.
- `MiniMax H3 Temporal RoPE Patch`: An experimental MODEL patch that pulls the target video token temporal RoPE toward one selected frame. Use it when trying to make the output more still-image-like.
- `Empty MiniMax H3 Single Frame Latent`: Creates an empty MiniMax H3-compatible AV latent for custom workflows.
- `MiniMax H3 Select Frame`: Selects one image from the decoded image sequence. The default frame index is `0`.

### Frame Count

For MiniMax H3 compatibility, the requested `frame_count` is rounded up until it satisfies:

```text
frame_count % 17 == 5
```

Examples:

```text
5  -> 5
6  -> 22
21 -> 22
22 -> 22
23 -> 39
```

The minimum is `5`. To get fewer output images, generate 5 frames and extract the needed frame with `MiniMax H3 Select Frame`.

### Resize

Input images are stretched internally to the requested size. There is no `resize mode` UI.

### Temporal RoPE Patch

`MiniMax H3 Temporal RoPE Patch` takes a MODEL and returns a MODEL. Connect it after the UNET Loader and before KSampler.

- `frame_index`: The latent frame used as the temporal RoPE reference. The default is `0`.
- `strength`: How strongly the normal temporal RoPE is pulled toward the fixed frame. The default is `1.0`, which fully freezes it to the selected frame. `0.5` pulls it halfway toward the selected frame, while `0.0` matches the behavior before this RoPE patch.

The default `frame_index = 0` and `strength = 1.0` is intended for still-image-like output. It may reduce motion, color shifts, and temporal drift. If the effect is too strong or causes artifacts, lower `strength` into the `0.25` to `0.75` range.

This is experimental. Because it changes the temporal position layout the model was trained with, it may also introduce smearing, repeated patterns, texture degradation, or composition errors. If it does not help, set `strength = 0.0`.

### Example Workflows

- `example_workflows/minimax_h3_single_frame_edit_basic.json`
- `example_workflows/minimax_h3_start_end_interpolate_middle.json`

Open a workflow in ComfyUI and replace the placeholder UNET, CLIP, VAE, and input image filenames with your local MiniMax H3 files.
The `sample` directory contains example input images for checking the workflows.

This custom node does not patch the MiniMax H3 core model code.
