"""
학습 상태 확인 스크립트
인터넷 연결 없이도 로컬에서 학습 진행 상황을 확인할 수 있습니다.
"""

import os
from pathlib import Path
import time

def check_training_status():
    """학습 상태 확인"""
    print("="*60)
    print("학습 상태 확인")
    print("="*60)
    
    # 1. 체크포인트 확인
    checkpoint_dir = Path("checkpoints")
    checkpoints = list(checkpoint_dir.glob("*.pth"))
    
    if checkpoints:
        print(f"\n[체크포인트] 발견: {len(checkpoints)}개")
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        print(f"   최신 파일: {latest.name}")
        print(f"   수정 시간: {time.ctime(latest.stat().st_mtime)}")
    else:
        print("\n[체크포인트] 없음 (학습이 아직 시작되지 않았거나 진행 중)")
    
    # 2. 로그 디렉토리 확인
    log_dir = Path("logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*"))
        if log_files:
            print(f"\n[로그 파일] {len(log_files)}개")
            for log_file in sorted(log_files)[-5:]:
                print(f"   - {log_file.name}")
    
    # 3. Python 프로세스 확인
    try:
        import psutil
        python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']) 
                           if 'python' in p.info['name'].lower()]
        
        training_processes = [p for p in python_processes 
                             if 'train.py' in ' '.join(p.cmdline())]
        
        if training_processes:
            print(f"\n[학습 프로세스] 실행 중: {len(training_processes)}개")
            for p in training_processes:
                print(f"   PID: {p.info['pid']}, CPU: {p.info['cpu_percent']:.1f}%, 메모리: {p.info['memory_info'].rss / 1024 / 1024:.1f}MB")
        else:
            print("\n[학습 프로세스] 실행 중이지 않음")
    except ImportError:
        print("\n[프로세스 확인] psutil 미설치")
        print("   설치: pip install psutil")
    
    # 4. 데이터셋 확인
    train_noisy = Path("data/train/noisy")
    if train_noisy.exists():
        train_files = list(train_noisy.glob("*.wav"))
        print(f"\n[학습 데이터] {len(train_files)}개 파일")
    
    val_noisy = Path("data/val/noisy")
    if val_noisy.exists():
        val_files = list(val_noisy.glob("*.wav"))
        print(f"[검증 데이터] {len(val_files)}개 파일")
    
    print("\n" + "="*60)
    print("학습 시작 방법:")
    print("  python training/train.py --config config.yaml")
    print("="*60)

if __name__ == "__main__":
    check_training_status()

