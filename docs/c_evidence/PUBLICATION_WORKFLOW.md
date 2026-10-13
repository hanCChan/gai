# 每轮代码与结果发布流程

发布由用户明确授权，目标仓库`hanCChan/gai`，使用账号hanCChan。
HTTPS命令行本机无凭据，使用已连接GitHub应用；不读取/输出凭据。

1. 本地提交完成的代码、计划、报告、决策、输入/代码SHA和证据索引。
2. 读取目标远端分支头，准备精确Git文件树：

   ```bash
   python scripts/prepare_research_publication.py --base <已核验的远端commit> \
     --output outputs/publication/unique_plan.json
   ```

3. 连接工具逐项create_blob(base64)，返回SHA必须与本地Git blob SHA一致。
   大文件通过helper的`--blob SHA --offset N --chunk 16000`读取，禁止把文件内容
   插入shell命令或输出到用户对话。相同blob只上传一次。
4. create_tree(base_tree_sha)，树SHA必须等于manifest.expected_tree。
5. create_commit，以已核验远端头为parent；新分支create_branch。已有分支必须
   先查当前头，普通update_ref，不force、不合并main、不覆盖其他实验分支。
6. git fetch该分支，比较文件树必须为空差分；用连接回读报告/决策文件，
   SHA必须与本地一致。随后交付可直接打开的GitHub链接。

连接API提交身份可能与本地commit不同，不伪称远端存在本地SHA。文件树和
内容一致性必须逐次确认；如冻结来源需要原始commit对象，另外发布精确
Git history bundle（不含私有/原始数据）并注明其SHA。

若上传失败，明确说“尚未上传/部分上传”及实际原因，不能只给本地路径。
不上传原稿、拒稿信、原始IQ、checkpoint或秘密文件。
