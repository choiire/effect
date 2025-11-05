"""
데이터 설정 검증 스크립트

학습 시작 전에 모든 데이터가 올바르게 준비되었는지 확인합니다.
"""

from pathlib import Path
from collections import defaultdict


def count_files(directory, pattern="*.wav"):
    """디렉토리의 파일 수 세기"""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def check_file_matching(noisy_dir, clean_dir):
    """noisy와 clean 파일이 올바르게 매칭되는지 확인"""
    noisy_path = Path(noisy_dir)
    clean_path = Path(clean_dir)
    
    if not noisy_path.exists() or not clean_path.exists():
        return False, []
    
    mismatches = []
    
    for noisy_file in noisy_path.glob("*.wav"):
        # xxx_noisy.wav -> xxx_clean.wav
        clean_name = noisy_file.stem.replace("_noisy", "_clean") + ".wav"
        expected_clean = clean_path / clean_name
        
        if not expected_clean.exists():
            mismatches.append(f"{noisy_file.name} -> {clean_name} (없음)")
    
    return len(mismatches) == 0, mismatches


def verify_data_setup():
    """전체 데이터 설정 검증"""
    
    print("=" * 70)
    print("데이터 설정 검증 (Data Setup Verification)")
    print("=" * 70)
    
    project_root = Path(__file__).parent
    
    # 검증 결과 저장
    results = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    all_passed = True
    
    # 1. Train 데이터 검증
    print("\n📦 [1/3] Train 데이터 검증")
    print("-" * 70)
    
    train_clean_count = count_files(project_root / "data/train/clean")
    train_noisy_count = count_files(project_root / "data/train/noisy")
    
    results['train']['clean_count'] = train_clean_count
    results['train']['noisy_count'] = train_noisy_count
    
    print(f"  Clean 파일: {train_clean_count:,}개")
    print(f"  Noisy 파일: {train_noisy_count:,}개")
    
    if train_clean_count == 0:
        print("  ❌ 에러: Train clean 파일이 없습니다!")
        print("     → data/train/clean/ 폴더에 음성 파일을 배치하세요")
        all_passed = False
    elif train_noisy_count == 0:
        print("  ❌ 에러: Train noisy 파일이 없습니다!")
        print("     → python prepare_training_data.py 를 실행하세요")
        all_passed = False
    elif train_clean_count != train_noisy_count:
        print(f"  ⚠️  경고: Clean과 Noisy 파일 수가 다릅니다!")
        all_passed = False
    else:
        # 파일 매칭 확인
        matched, mismatches = check_file_matching(
            project_root / "data/train/noisy",
            project_root / "data/train/clean"
        )
        
        if matched:
            print(f"  ✅ Train 데이터 준비 완료 ({train_clean_count:,}개 쌍)")
        else:
            print(f"  ❌ 에러: {len(mismatches)}개 파일이 매칭되지 않습니다")
            if len(mismatches) <= 5:
                for mismatch in mismatches:
                    print(f"     - {mismatch}")
            all_passed = False
    
    # 2. Val 데이터 검증
    print("\n📦 [2/3] Validation 데이터 검증")
    print("-" * 70)
    
    val_clean_count = count_files(project_root / "data/val/clean")
    val_noisy_count = count_files(project_root / "data/val/noisy")
    
    results['val']['clean_count'] = val_clean_count
    results['val']['noisy_count'] = val_noisy_count
    
    print(f"  Clean 파일: {val_clean_count:,}개")
    print(f"  Noisy 파일: {val_noisy_count:,}개")
    
    if val_clean_count == 0:
        print("  ❌ 에러: Val clean 파일이 없습니다!")
        print("     → data/val/clean/ 폴더에 음성 파일을 배치하세요")
        all_passed = False
    elif val_noisy_count == 0:
        print("  ❌ 에러: Val noisy 파일이 없습니다!")
        print("     → python prepare_validation_data.py 를 실행하세요")
        all_passed = False
    elif val_clean_count != val_noisy_count:
        print(f"  ⚠️  경고: Clean과 Noisy 파일 수가 다릅니다!")
        all_passed = False
    else:
        matched, mismatches = check_file_matching(
            project_root / "data/val/noisy",
            project_root / "data/val/clean"
        )
        
        if matched:
            print(f"  ✅ Val 데이터 준비 완료 ({val_clean_count:,}개 쌍)")
        else:
            print(f"  ❌ 에러: {len(mismatches)}개 파일이 매칭되지 않습니다")
            all_passed = False
    
    # 3. Test 데이터 검증 (선택적)
    print("\n📦 [3/3] Test 데이터 검증 (선택적)")
    print("-" * 70)
    
    test_clean_count = count_files(project_root / "data/test/clean")
    test_noisy_count = count_files(project_root / "data/test/noisy")
    
    results['test']['clean_count'] = test_clean_count
    results['test']['noisy_count'] = test_noisy_count
    
    print(f"  Clean 파일: {test_clean_count:,}개")
    print(f"  Noisy 파일: {test_noisy_count:,}개")
    
    if test_clean_count == 0:
        print("  ℹ️  정보: Test clean 파일이 없습니다 (선택적)")
    elif test_noisy_count == 0:
        print("  ⚠️  경고: Test clean은 있지만 noisy가 없습니다")
        print("     → python prepare_test_data.py 를 실행하세요 (평가 시)")
    elif test_clean_count != test_noisy_count:
        print(f"  ⚠️  경고: Clean과 Noisy 파일 수가 다릅니다!")
    else:
        matched, mismatches = check_file_matching(
            project_root / "data/test/noisy",
            project_root / "data/test/clean"
        )
        
        if matched:
            print(f"  ✅ Test 데이터 준비 완료 ({test_clean_count:,}개 쌍)")
            print(f"  ⚠️  주의: Test 데이터는 학습에 사용되지 않습니다!")
        else:
            print(f"  ❌ 에러: {len(mismatches)}개 파일이 매칭되지 않습니다")
    
    # 최종 요약
    print("\n" + "=" * 70)
    print("검증 요약 (Verification Summary)")
    print("=" * 70)
    
    print(f"\n📊 데이터 통계:")
    print(f"  Train: {train_clean_count:,}개 (학습용 ✅)")
    print(f"  Val:   {val_clean_count:,}개 (검증용 ✅)")
    print(f"  Test:  {test_clean_count:,}개 (평가용 ⚠️ 학습 X)")
    
    total_training_samples = train_clean_count + val_clean_count
    print(f"\n  학습에 사용되는 총 샘플: {total_training_samples:,}개")
    
    # 데이터 분리 확인
    print(f"\n🔒 데이터 분리 검증:")
    if test_clean_count > 0:
        print(f"  ✅ Test 데이터가 별도로 준비되어 있습니다")
        print(f"  ⚠️  Test 데이터는 절대 학습에 사용하지 마세요!")
    else:
        print(f"  ℹ️  Test 데이터 없음 (평가 시 준비)")
    
    # 최종 판정
    print("\n" + "=" * 70)
    if all_passed and train_clean_count > 0 and val_clean_count > 0:
        print("✅ 모든 검증 통과! 학습을 시작할 수 있습니다.")
        print("=" * 70)
        print("\n다음 명령으로 학습 시작:")
        print("  python training/train.py")
    else:
        print("❌ 일부 검증 실패. 위의 에러를 수정하세요.")
        print("=" * 70)
        print("\n문제 해결 방법:")
        
        if train_clean_count == 0:
            print("  1. data/train/clean/ 폴더에 음성 파일 배치")
        if train_noisy_count == 0:
            print("  2. python prepare_training_data.py 실행")
        if val_clean_count == 0:
            print("  3. data/val/clean/ 폴더에 음성 파일 배치")
        if val_noisy_count == 0:
            print("  4. python prepare_validation_data.py 실행")
    
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    verify_data_setup()

