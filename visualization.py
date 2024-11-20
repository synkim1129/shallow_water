import get_param
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from Logger import Logger
from pde_cnn import WaterSurfaceUNet
from setups import WaterSurfaceDataset
from derivatives import toCuda, toCpu
import time
import math

def vector2HSV(vector,plot_sqrt=False):
	"""
	transform vector field into hsv color wheel
	:vector: vector field (size: 2 x height x width)
	:return: hsv (hue: direction of vector; saturation: 1; value: abs value of vector)
	"""
	
	values = torch.sqrt(torch.sum(torch.pow(vector,2),dim=0)).unsqueeze(0)
	saturation = torch.ones(values.shape).cuda()
	norm = vector/(values+0.000001)
	angles = torch.asin(norm[0])+math.pi/2
	angles[norm[1]<0] = 2*math.pi-angles[norm[1]<0]
	hue = angles.unsqueeze(0)/(2*math.pi)
	hue = (hue*360+100)%360
	#values = norm*torch.log(values+1)
	values = values/torch.max(values)
	if plot_sqrt:
		values = torch.sqrt(values)
	hsv = torch.cat([hue,saturation,values])
	return hsv.permute(1,2,0).cpu().numpy()

# 설정
torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

params = get_param.params()

# 모델 로드
logger = Logger(get_param.get_hyperparam(params), use_tensorboard=False)
fluid_model = toCuda(WaterSurfaceUNet(params.hidden_size))
date_time, index = logger.load_state(fluid_model, None, 
								   datetime=params.load_date_time, 
								   index=params.load_index)
fluid_model.eval()
print(f"Loaded model: {date_time}, index: {index}")

# OpenCV 윈도우 설정
cv2.namedWindow('Water Height', cv2.WINDOW_NORMAL)
cv2.namedWindow('Velocity Field', cv2.WINDOW_NORMAL)

# 동영상 저장 설정
save_movie = False
if save_movie:
	fourcc = cv2.VideoWriter_fourcc(*'DIVX')
	movie_h = cv2.VideoWriter(f'plots/height_{get_param.get_hyperparam(params)}.avi', 
							 fourcc, 20.0, (params.width, params.height))
	movie_v = cv2.VideoWriter(f'plots/velocity_{get_param.get_hyperparam(params)}.avi', 
							 fourcc, 20.0, (params.width, params.height))

# 데이터셋 초기화 (여러 배치 미리 로드)
dataset = WaterSurfaceDataset(
	width=params.width,
	height=params.height,
	batch_size=100,
	dataset_size=100,
	average_sequence_length=params.average_sequence_length,
	H0=params.H0,
	dt=params.dt,
	boundary_type=params.boundary_type
)

# setup opencv windows:
cv2.namedWindow('legend',cv2.WINDOW_NORMAL) # legend for velocity field
vector = torch.cat([torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(2).repeat(1,1,200),torch.arange(-1,1,0.01).unsqueeze(0).unsqueeze(1).repeat(1,200,1)]).cuda()
image = vector2HSV(vector)
image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
cv2.imshow('legend',image)

# 시뮬레이션 실행
paused = False
quit_sim = False

# 시뮬레이션 실행
paused = False
quit_sim = False
current_batch = 0

with torch.no_grad():
	while not quit_sim:  # 전체 시뮬레이션 루프
		FPS = 0
		FPS_Counter = 0
		last_time = time.time()
		
		# 현재 배치의 초기 상태 가져오기
		h_delta, u, v, boundary_mask, debug_info = dataset.ask(debug=True)
		# import pdb; pdb.set_trace()
		h_delta = toCuda(h_delta)
		u = toCuda(u)
		v = toCuda(v)
		boundary_mask = toCuda(boundary_mask)
		
		h_delta_current = h_delta[current_batch:current_batch+1]
		u_current = u[current_batch:current_batch+1]
		v_current = v[current_batch:current_batch+1]
		boundary_mask_current = boundary_mask[current_batch:current_batch+1]
		debug_info_current = debug_info[current_batch]
		print(f"Initial State: {debug_info_current}")
		
		# 초기 상태 시각화
		print(f"Batch: {current_batch}, Step: 0 (Initial State)")
		
		# 수심 시각화
		h = toCpu(h_delta_current[0, 0] + params.H0).numpy()
		h_normalized = (h - np.min(h)) / (np.max(h) - np.min(h))
		h_colored = cv2.cvtColor((plt.cm.cool(h_normalized)[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
		
		# # 속도장 시각화
		# import pdb; pdb.set_trace()
		u_field = u_current[0]
		v_field = v_current[0]
		velocity_vector = torch.cat([u_field, v_field], dim=0)
		vel_vis = vector2HSV(velocity_vector)
		vel_vis = cv2.cvtColor((vel_vis * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
		
		# 결과 표시
		cv2.imshow('Water Height', h_colored)
		cv2.imshow('Velocity Field', vel_vis)
		
		# 동영상 저장
		if save_movie:
			movie_h.write(h_colored)
			movie_v.write(vel_vis)
		
		# 단일 초기조건에 대한 시뮬레이션 루프
		for t in range(params.average_sequence_length):
			if not paused:
				# 모델로 다음 상태 예측
				time.sleep(params.dt/2)
				h_delta_new, u_new, v_new = fluid_model(h_delta_current, u_current, v_current)
				
				if t % 1 == 0:  # 10 스텝마다 시각화
					print(f"Batch: {current_batch}, Step: {t+1}")  # t+1로 변경
					
					# 수심 시각화
					h = toCpu(h_delta_new[0, 0] + params.H0).numpy()
					h_normalized = (h - np.min(h)) / (np.max(h) - np.min(h))
					h_colored = cv2.cvtColor((plt.cm.cool(h_normalized)[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
					
					# 속도장 시각화
					u_field = u_new[0]
					v_field = v_new[0]
					velocity_vector = torch.cat([u_field, v_field], dim=0)
					vel_vis = vector2HSV(velocity_vector)
					vel_vis = cv2.cvtColor((vel_vis * 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
					
					# 결과 표시
					cv2.imshow('Water Height', h_colored)
					cv2.imshow('Velocity Field', vel_vis)
					
					# 동영상 저장
					if save_movie:
						movie_h.write(h_colored)
						movie_v.write(vel_vis)
					
					# FPS 계산
					FPS_Counter += 1
					if time.time() - last_time >= 1:
						FPS = FPS_Counter
						FPS_Counter = 0
						last_time = time.time()
				
				# 현재 상태 업데이트
				h_delta_current = h_delta_new
				u_current = u_new
				v_current = v_new
			
			# 키보드 입력 처리
			key = cv2.waitKey(1)
			if key == ord('q'):  # 종료
				quit_sim = True
				break
			elif key == ord('n'):  # 다음 배치로
				current_batch = (current_batch + 1) % dataset.batch_size
				break
			elif key == ord(' '):  # 일시정지/재개
				paused = not paused
				status = "Paused" if paused else "Running"
				print(f"Simulation {status}")

# 정리
if save_movie:
	movie_h.release()
	movie_v.release()

cv2.destroyAllWindows()