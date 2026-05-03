import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.environ.get("SUPABASE_KEY", "").strip()
BUCKET_NAME = "vocab-audio"
LOCAL_FOLDER = "/Users/chenchanghou/Documents/trae_projects/n1_work/vocab-audio-files"

# 初始化客户端
if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("缺少 SUPABASE_URL 或 SUPABASE_SERVICE_ROLE_KEY 环境变量")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def resume_upload():
    # 1. 分页获取云端已有文件，确保不重复上传
    print("正在连接 Supabase 获取已上传列表，请稍候...")
    remote_files = set()
    offset = 0
    while True:
        try:
            res = supabase.storage.from_(BUCKET_NAME).list("", {"limit": 1000, "offset": offset})
            if not res:
                break
            names = [f['name'] for f in res if f['name'] != '.emptyFolderPlaceholder']
            remote_files.update(names)
            if len(names) < 1000:
                break
            offset += 1000
        except Exception as e:
            print(f"获取列表时出错: {e}")
            break
    
    print(f"✅ 云端检查完毕：已存在 {len(remote_files)} 个文件。")

    # 2. 读取本地文件夹中的 MP3
    if not os.path.exists(LOCAL_FOLDER):
        print(f"❌ 错误：找不到本地文件夹 {LOCAL_FOLDER}")
        return

    local_files = [f for f in os.listdir(LOCAL_FOLDER) if f.endswith('.mp3')]
    
    # 过滤出还没上传的文件
    to_upload = [f for f in local_files if f not in remote_files]
    
    print(f"📊 统计信息：")
    print(f"   - 本地总计: {len(local_files)} 个")
    print(f"   - 待补传: {len(to_upload)} 个")

    if not to_upload:
        print("🎉 太棒了！所有文件都已上传成功，无需补传。")
        return

    print("🚀 开始补传...")

    # 3. 执行补传逻辑
    success_count = 0
    error_count = 0

    for i, file_name in enumerate(to_upload, 1):
        file_path = os.path.join(LOCAL_FOLDER, file_name)
        print(f"[{i}/{len(to_upload)}] 正在上传: {file_name}", end=" ... ")
        
        try:
            with open(file_path, 'rb') as f:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=file_name, 
                    file=f,
                    file_options={"content-type": "audio/mpeg"}
                )
            print("完成 ✅")
            success_count += 1
        except Exception as e:
            # 再次检查是否是因为已存在导致的报错
            if "already exists" in str(e):
                print("跳过（已存在）")
                success_count += 1
            else:
                print(f"失败 ❌ ({str(e)})")
                error_count += 1

    print(f"\n--- 任务总结 ---")
    print(f"成功: {success_count} 个")
    print(f"失败: {error_count} 个")
    if error_count == 0:
        print("所有缺失文件已补齐！")

if __name__ == "__main__":
    resume_upload()
