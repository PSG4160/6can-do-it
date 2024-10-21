import random

def num_guess():
    # 1과 10 사이의 랜덤 숫자를 생성
    num = random.randint(1, 10)
    print("1과 10 사이의 숫자를 하나 정했습니다.")
    
    while True:  # 무한 루프 시작
        try:
            # 사용자로부터 숫자 입력 받기
            guessed_num = int(input("예상 숫자: "))
            # 입력된 숫자가 유효한 범위인지 확인
            if guessed_num < 1 or guessed_num > 10:
                print("유효한 범위 내의 숫자를 다시 입력하세요.")
            elif guessed_num < num:
                print("너무 작습니다. 다시 입력하세요.")
            elif guessed_num > num:
                print("너무 큽니다. 다시 입력하세요.")
            else:
                print("정답입니다!")  # 정답을 맞춘 경우
                break  # 루프 종료
        except ValueError as e:
            # 숫자가 아닌 값을 입력했을 때의 처리
            print(f"입력 오류: {e}. 유효한 숫자를 입력하세요.")

# 게임 시작
num_guess()

while True:  # 게임 재시작 여부 확인
    restart = input("게임을 다시 하시곗습니까? (y/n): ").strip().lower()
    if restart == 'y':  # 'y' 입력 시 게임 재시작
        num_guess()
    elif restart == 'n':  # 'n' 입력 시 게임 종료
        print("게임을 종료합니다. 감사합니다!")
        break  # 루프 종료
    else:
        continue  # 유효하지 않은 입력 시 다시 확인
