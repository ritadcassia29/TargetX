# drone_control.py

import time
from pymavlink import mavutil
import math
import pymap3d.enu as enu
import pymap3d as pm

delay_comando = 0.5

def distance_between_geo_points(point1, point2):
    """
    Calcula a distância entre dois pontos geográficos (latitude, longitude, altitude)
    usando o sistema de coordenadas locais ENU (East‑North‑Up) via pymap3d.

    Parâmetros:
        lat1, lon1: latitude e longitude do ponto de referência (graus decimais)
        lat2, lon2: latitude e longitude do ponto alvo (graus decimais)
        alt1: altitude do ponto de referência (metros) — default 0.0
        alt2: altitude do ponto alvo (metros) — default 0.0

    Retorna:
        Distância em metros (float)
    """
    lat1, lon1, alt1 = point1
    lat2, lon2, alt2 = point2
    # Converte ponto alvo para coordenadas ENU, com origem no ponto 1
    e, n, u = enu.geodetic2enu(lat2, lon2, alt2, lat1, lon1, alt1)

    # Distância euclidiana no sistema ENU
    return math.sqrt(e**2 + n**2 + u**2)

# Função para conectar ao drone
def connect_drone_IP(connection_string):

    drone = mavutil.mavlink_connection(connection_string)
    drone.wait_heartbeat()

    print("Enxame conectado!")

    return drone

def connect_drone(port, baud):

    drone = mavutil.mavlink_connection(port, baud=baud)
    drone.wait_heartbeat()

    print("Conectado ao drone!")

    return drone



def connect_drone_sitl(host="127.0.0.1", port=5762):
    """
    Conecta ao SITL ArduCopter via TCP.
    host: IP do SITL (normalmente localhost)
    port: porta SERIAL0 do SITL (5760 por padrão)
    """
    connection_string = f"tcp:{host}:{port}"
    print(f"Conectando ao SITL em {connection_string}...")
    drone = mavutil.mavlink_connection(connection_string)
    drone.wait_heartbeat()

    print("Conectado ao SITL!")

    return drone

# Função para definir modo de voo
def set_flight_mode(drone, mode_name):

    id_drone = drone.target_system
    
    print(f"Drone {id_drone}: Alterando para modo {mode_name}...")

    while drone.flightmode != mode_name:

        drone.wait_heartbeat()

        mode_id = drone.mode_mapping()[mode_name]

        drone.mav.set_mode_send(
            id_drone,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        
        time.sleep(delay_comando)
    
    print(f"Drone {id_drone}: Modo alterado para {mode_name}")

def set_param(drone, name: str, value: float,
              param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
              timeout=3.0) -> bool:
    """
    Seta um parâmetro ArduPilot e aguarda confirmação (compatível com
    pymavlink antigo e novo).
    """
    drone.mav.param_set_send(
        drone.target_system,
        drone.target_component,
        name.encode("utf-8"),
        float(value),
        param_type
    )

    t0 = time.time()

    while time.time() - t0 < timeout:
        msg = drone.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
        if not msg:
            continue

        param_id = msg.param_id
        if isinstance(param_id, bytes):
            param_id = param_id.decode("utf-8")

        param_id = param_id.strip("\x00")

        if param_id == name:
            print(f"Drone {drone.target_system}: {name} confirmado = {msg.param_value}")
            return True

    print(f"Drone {drone.target_system}: não confirmou {name}, seguindo mesmo assim...")
    
    return False

def set_param_async(drone, name: str, value: float,
              param_type=mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
              timeout=3.0, tentativas=3) -> None:
    """
    Seta um parâmetro ArduPilot e aguarda confirmação (compatível com
    pymavlink antigo e novo).
    """
    for i in range(tentativas):

        drone.mav.param_set_send(
            drone.target_system,
            drone.target_component,
            name.encode("utf-8"),
            float(value),
            param_type
        )

        time.sleep(0.2)



def movimento_relativo(drone, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):


    drone.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms
        drone.target_system,
        drone.target_component,

        mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,  # <<<<< referencial global
        # mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # <<<<< referencial no veiculo

        0b110111111000,

        dx,    # x: 2 metros para frente (eixo X do nariz)
        dy,      # y: 0 (não lateral)
        dz,      # z: 0 (mantém altitude)

        0, 0, 0,  # vx, vy, vz (ignorados)
        0, 0, 0,  # afx, afy, afz (ignorados)
        0, 0)    # yaw, yaw_rate (ignorados)


def goto_location(drone, point, altitude):
    # Desempacota as coordenadas do ponto (lat, lon)
    lat, lon = point

    id_drone = drone.target_system

    vel_x = 0
    vel_y = 0
    vel_z = 0

    graus_rate = 0
    yaw_rate = graus_rate * 2 * math.pi / 360

    # Obtém a altitude atual do drone
    # current_altitude = get_altitude(drone)
    
    print(f"Drone {id_drone}: Indo para lat: {lat}, lon: {lon}, altitude: {altitude}m...")

    # Bitmask to indicate which fields should be ignored by the vehicle (see POSITION_TARGET_TYPEMASK enum)
    # bit1:PosX, bit2:PosY, bit3:PosZ, bit4:VelX, bit5:VelY, bit6:VelZ, bit7:AccX, bit8:AccY, bit9:AccZ, bit11:yaw, bit12:yaw rate
    # When providing Pos, Vel and/or Accel all 3 axis must be provided. At least one of Pos, Vel and Accel must be provided (e.g. providing Yaw or YawRate alone is not supported)
    # Use Position : 0b110111111000 / 0x0DF8 / 3576 (decimal)
    # Use Velocity : 0b110111000111 / 0x0DC7 / 3527 (decimal)
    # Use Acceleration : 0b110000111000 / 0x0C38 / 3128 (decimal)
    # Use Pos+Vel : 0b110111000000 / 0x0DC0 / 3520 (decimal)
    # Use Pos+Vel+Accel : 0b110000000000 / 0x0C00 / 3072 (decimal)
    # Use Yaw : 0b100111111111 / 0x09FF / 2559 (decimal)
    # Use Yaw Rate : 0b010111111111 / 0x05FF / 1535 (decimal)

    	
    # Valid options are:
    # MAV_FRAME_GLOBAL (0): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_INT (5): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_RELATIVE_ALT (3): alt is in meters above home
    # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT (6): alt is in meters above home
    # MAV_FRAME_GLOBAL_TERRAIN_ALT (10): alt is in meters above terrain
    # MAV_FRAME_GLOBAL_TERRAIN_ALT_INT (11): alt is in meters above terrain

    tentativas_envio = 3

    for i in range(tentativas_envio):

        drone.mav.set_position_target_global_int_send(
            0, id_drone, drone.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            #   0b110111000000,  #Posição + velocidade
            # 0b110111000000 
            0b110111111000,  # Máscara para posição
            int(lat * 10**7), 
            int(lon * 10**7), 
            altitude,
            vel_x, 
            vel_y, 
            vel_z, 
            0, 
            0, 
            0, 
            0, 
            yaw_rate)
        
        time.sleep(0.2)





def goto_location_and_wait(drone, point, altitude, erro_ate_ponto=5):
    # Desempacota as coordenadas do ponto (lat, lon)
    lat, lon = point

    id_drone = drone.target_system

    vel_x = 0
    vel_y = 0
    vel_z = 0

    graus_rate = 0
    yaw_rate = graus_rate * 2 * math.pi / 360

    # Obtém a altitude atual do drone
    # current_altitude = get_altitude(drone)
    
    print(f"Drone {id_drone}: Indo para lat: {lat}, lon: {lon}, altitude: {altitude}m...")

    # Bitmask to indicate which fields should be ignored by the vehicle (see POSITION_TARGET_TYPEMASK enum)
    # bit1:PosX, bit2:PosY, bit3:PosZ, bit4:VelX, bit5:VelY, bit6:VelZ, bit7:AccX, bit8:AccY, bit9:AccZ, bit11:yaw, bit12:yaw rate
    # When providing Pos, Vel and/or Accel all 3 axis must be provided. At least one of Pos, Vel and Accel must be provided (e.g. providing Yaw or YawRate alone is not supported)
    # Use Position : 0b110111111000 / 0x0DF8 / 3576 (decimal)
    # Use Velocity : 0b110111000111 / 0x0DC7 / 3527 (decimal)
    # Use Acceleration : 0b110000111000 / 0x0C38 / 3128 (decimal)
    # Use Pos+Vel : 0b110111000000 / 0x0DC0 / 3520 (decimal)
    # Use Pos+Vel+Accel : 0b110000000000 / 0x0C00 / 3072 (decimal)
    # Use Yaw : 0b100111111111 / 0x09FF / 2559 (decimal)
    # Use Yaw Rate : 0b010111111111 / 0x05FF / 1535 (decimal)

    	
    # Valid options are:
    # MAV_FRAME_GLOBAL (0): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_INT (5): alt is in meters above sea level
    # MAV_FRAME_GLOBAL_RELATIVE_ALT (3): alt is in meters above home
    # MAV_FRAME_GLOBAL_RELATIVE_ALT_INT (6): alt is in meters above home
    # MAV_FRAME_GLOBAL_TERRAIN_ALT (10): alt is in meters above terrain
    # MAV_FRAME_GLOBAL_TERRAIN_ALT_INT (11): alt is in meters above terrain

    tentativas_envio = 3

    for i in range(tentativas_envio):

        drone.mav.set_position_target_global_int_send(
            0, id_drone, drone.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            #   0b110111000000,  #Posição + velocidade
            # 0b110111000000 
            0b110111111000,  # Máscara para posição
            int(lat * 10**7), 
            int(lon * 10**7), 
            altitude,
            vel_x, 
            vel_y, 
            vel_z, 
            0, 
            0, 
            0, 
            0, 
            yaw_rate)
        
        time.sleep(0.2)


    
    while True:

        distancia_ate_ponto = distancia_ate(drone, point)

        if distancia_ate_ponto is not None:
            if distancia_ate_ponto < erro_ate_ponto :
                print(f"Drone {id_drone}: chegou")
                break

        time.sleep(0.1)

        # else:
        #     print(f"Drone {id_drone}: distancia até ponto {distancia_ate_ponto}")


def enviar_velocidade_visual(drone, velocidade_x, velocidade_y):
    """
    Envia velocidade para o drone usando controle visual.

    velocidade_x:
        velocidade no eixo longitudinal do drone (frente/trás)

    velocidade_y:
        velocidade lateral do drone (direita/esquerda)
    """

    velocidade_forward = velocidade_x
    velocidade_right = velocidade_y
    velocidade_down = 0.0

    drone.mav.set_position_target_local_ned_send(
        0,
        drone.target_system,
        drone.target_component,

        mavutil.mavlink.MAV_FRAME_BODY_NED,

        # Usar somente velocidade
        0b110111000111,

        # Posição ignorada
        0, 0, 0,

        # Velocidade
        velocidade_forward,
        velocidade_right,
        velocidade_down,

        # Aceleração ignorada
        0, 0, 0,

        # Yaw e yaw rate ignorados
        0,
        0
    )
    
def calcular_bearing_old(lat1, lon1, lat2, lon2):
    """
    Retorna o bearing (azimute) em graus [0,360),
    do ponto (lat1, lon1) para (lat2, lon2).
    """
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)

    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0

def calcular_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Retorna o bearing (azimute) em graus [0,360),
    do ponto (lat1, lon1) para (lat2, lon2).
    """

    alt_m = 0

    # Converte destino para coordenadas locais ENU, com origem em (lat1, lon1, alt_m)
    e, n, _u = pm.geodetic2enu(lat2, lon2, alt_m, lat1, lon1, alt_m, deg=True)

    # Bearing no plano: atan2(E, N) (note a ordem!)
    bearing = math.degrees(math.atan2(e, n))

    return (bearing + 360.0) % 360.0

def apontar_yaw_para_waypoint( drone, waypoint_desejado: tuple, yaw_rate_deg_s: float = 20.0, sentido_horario: bool = True, relativo: bool = False):
    """
    Controla APENAS o yaw do drone para apontar para um waypoint.

    Parâmetros:
        drone: conexão pymavlink
        waypoint: (lat, lon)
        yaw_rate_deg_s: velocidade de giro (graus/s)
        sentido_horario: True = horário, False = anti-horário
        relativo: False = yaw absoluto (recomendado)
    """

    current_lat = None
    current_long = None
    yaw_atual = None

    tentativas_leitura = 3

    for i in range(tentativas_leitura):

        msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        
        if msg is not None:
            if msg.get_type() == 'GLOBAL_POSITION_INT':
                # Posição absoluta em milímetros e velocidade em cm/s
                current_lat = msg.lat / 1e7          # graus
                current_long = msg.lon / 1e7          # graus
                yaw_atual  = msg.hdg / 100.0        # graus (0-360)

                if(current_lat is not None) and (current_long is not None) and (yaw_atual is not None):
                    break


    if current_lat is None or current_long is None or yaw_atual is None:
        print(f"Drone {drone.target_system}: sem posição para yaw")
        return

    lat_wp, lon_wp = waypoint_desejado

    yaw_desejado = calcular_bearing(current_lat, current_long, lat_wp, lon_wp)

    # Diferença normalizada (-180 a +180)
    delta_yaw = (yaw_desejado - yaw_atual + 180.0) % 360.0 - 180.0

    # Define sentido do giro mínimo
    sentido = 1 if delta_yaw >= 0 else -1   # horário ou anti-horário

    
    tentativas_envio = 3

    for i in range(tentativas_envio):

        drone.mav.command_long_send(
            drone.target_system,
            drone.target_component,
            mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            0,
            yaw_desejado,                 # param1: yaw alvo (graus)
            yaw_rate_deg_s,               # param2: velocidade (graus/s)
            sentido,            # direção automática
            1 if relativo else 0,          # param4: 0 = absoluto, 1 = relativo
            0, 0, 0
        )

        time.sleep(0.2)

    print(f"Drone {drone.target_system}: yaw → {yaw_desejado:.1f}°")


def distancia_ate(drone, ponto):
    
    lat, lon = ponto

    tentativas = 5
    i=0

    while i<tentativas:

        i=i+1

        msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        
        if msg is not None:

            if msg.get_type() == 'GLOBAL_POSITION_INT':
                # Posição absoluta em milímetros e velocidade em cm/s
                current_lat = msg.lat / 1e7          # graus
                current_long = msg.lon / 1e7          # graus
                alt = msg.relative_alt / 1000.0  # metros (relativo ao home)
                vx = msg.vx / 100.0          # m/s
                vy = msg.vy / 100.0          # m/s
                vz = msg.vz / 100.0          # m/s
                hdg = msg.hdg / 100.0        # graus (0-360)

                # print(f"Pos: {lat:.7f}, {lon:.7f} | Alt: {alt:.2f}m | Vel: ({vx:.1f}, {vy:.1f}, {vz:.1f}) m/s | Hdg: {hdg:.1f}°")

                current_point = current_lat, current_long, 0
                dest_point = lat, lon, 0

                distance = distance_between_geo_points(current_point, dest_point)
                # voltagem_bateria = get_bateria_v(drone)

                # print(f"Distancia ate waypoint: {distance}")

                return distance
            
    id_drone = drone.target_system
    print(f"Drone {id_drone}: sem telemetria de distancia até waypoint")

    return None


def get_altitude_relativa_m(drone, timeout=0.2) -> float | None:
    """
    Lê a altitude relativa (m) usando GLOBAL_POSITION_INT, evitando ler dados antigos:
    drena a fila e retorna a última mensagem recebida.
    """
    last = None

    # 1) Aguarda pelo menos 1 mensagem (com timeout)
    msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=timeout)
    
    if msg is None:
        return None
    last = msg

    # 2) Drena rapidamente tudo que já está no buffer (não-bloqueante)
    while True:
        msg = drone.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg is None:
            break
        last = msg

    # relative_alt: mm -> m
    return last.relative_alt / 1000.0

def wait_until_altitude( drone, altitude_alvo_m: float, tolerancia_m: float = 0.5, timeout_s: float = 30.0, intervalo_s: float = 0.2) -> bool:
    """
    Aguarda até o drone atingir altitude_alvo_m (relativa ao HOME),
    dentro de uma tolerância, ou até estourar timeout.
    """
    t0 = time.time()

    while time.time() - t0 < timeout_s:

        alt = get_altitude_relativa_m(drone, timeout=1.0)
        if alt is None:
            continue

        erro = altitude_alvo_m - alt

        print(f"Drone {drone.target_system}: altura atual: {alt}  -  altura alvo: {altitude_alvo_m}")

        if abs(erro) <= tolerancia_m:
            return True

        # opcional: print leve
        # print(f"Drone {drone.target_system}: alt={alt:.2f}m alvo={altitude_alvo_m:.2f}m erro={erro:.2f}m")

        time.sleep(intervalo_s)

    return False


def lancar_carga_multiplos_drones(drones):
    """
    Aciona um servo na porta AUX3 da Pixhawk 6C em modo reverso.
    
    - ativado=True → envia PWM baixo (ativado)
    - ativado=False → envia PWM alto (desativado)
    - pwm_ativado=900 → valor de PWM para "ligado"
    - pwm_desativado=1900 → valor de PWM para "desligado"
    """

    # Porta AUX3 = canal 10 (1-8 = MAIN, 9-14 = AUX)
    canal = 11
    pwm_baixo = 1000
    pwm_alta = 2000
    tentativas = 3

    for i in range(0,tentativas):

        print(f"LAMÇAMENTO DE CARGA: Tentativa {i +1}")
    
        for drone in drones:
            drone.mav.command_long_send(
                drone.target_system,
                drone.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                canal,     # Número do servo
                pwm_baixo,       # PWM a enviar
                0, 0, 0, 0, 0
            )

        time.sleep(3)

        for drone in drones:
            drone.mav.command_long_send(
                drone.target_system,
                drone.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
                0,
                canal,     # Número do servo
                pwm_alta,       # PWM a enviar
                0, 0, 0, 0, 0
            )

        time.sleep(3)

def set_velocidade_drones(drones, velocidade):

    for drone in drones:
        set_param_async(drone, "WPNAV_SPEED", velocidade, tentativas=3)



def lancar_carga(drone):
    """
    Aciona um servo na porta AUX3 da Pixhawk 6C em modo reverso.
    
    - ativado=True → envia PWM baixo (ativado)
    - ativado=False → envia PWM alto (desativado)
    - pwm_ativado=900 → valor de PWM para "ligado"
    - pwm_desativado=1900 → valor de PWM para "desligado"
    """

    # Porta AUX3 = canal 10 (1-8 = MAIN, 9-14 = AUX)
    canal = 11
    pwm_baixo = 1000
    pwm_alta = 2000
    tentativas = 3

    for i in range(0,tentativas):

        print(f"LANÇAMENTO DE CARGA: Tentativa {i +1}")
    
       
        drone.mav.command_long_send(
            drone.target_system,
            drone.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0,
            canal,     # Número do servo
            pwm_baixo,       # PWM a enviar
            0, 0, 0, 0, 0
        )

        time.sleep(0.2)


        # drone.mav.command_long_send(
        #     drone.target_system,
        #     drone.target_component,
        #     mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        #     0,
        #     canal,     # Número do servo
        #     pwm_alta,       # PWM a enviar
        #     0, 0, 0, 0, 0
        # )

        # time.sleep(3)

def arm_and_takeoff(drone, altitude):
    
    id_drone = drone.target_system

    print(f"Drone {id_drone} - Armando e decolando a {altitude}m...")

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    
    time.sleep(3)

    msg = drone.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
    if msg:
        print(f"STATUSTEXT: {msg.text}")

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude)
    
    msg = drone.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
    if msg:
        print(f"STATUSTEXT: {msg.text}")

    
    time.sleep(delay_comando)


def arm_takeoff_and_wait(drone, altitude: float, tolerancia_m: float = 0.5, timeout_s: float = 40.0):
    
    id_drone = drone.target_system

    print(f"Drone {id_drone} - Armando e decolando a {altitude}m...")

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0, 1, 0, 0, 0, 0, 0, 0)
    
    time.sleep(3)

    msg = drone.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
    if msg:
        print(f"STATUSTEXT: {msg.text}")

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0, 0, 0, 0, 0, 0, 0, altitude)
    

    tolerancia_m: float = 0.5
    timeout_s: float = 40.0

    ok = wait_until_altitude(
            drone,
            altitude_alvo_m=altitude,
            tolerancia_m=tolerancia_m,
            timeout_s=timeout_s,
            intervalo_s=0.2
        )

    if ok:
        print(f"Drone {id_drone}: atingiu {altitude:.1f}m (±{tolerancia_m:.1f}m)")
    else:
        alt_atual = get_altitude_relativa_m(drone, timeout=1.0)
        print(f"Drone {id_drone}: TIMEOUT! alt atual={alt_atual}m alvo={altitude}m")

    
    time.sleep(delay_comando)

# Função para retornar e pousar
def return_and_land(drone):
    id_drone = drone.target_system

    print(f"Drone {id_drone} - Retornando ao ponto de origem e pousando...")

    set_flight_mode(drone, id_drone, "RTL")

    time.sleep(5)

    drone.mav.command_long_send(
        id_drone, drone.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    
    time.sleep(5)

def pressionar_para_continuar():
    """
    Pausa a execução até o usuário pressionar qualquer tecla.
    Funciona em Windows, Linux e macOS.
    """
    try:
        # Windows
        import msvcrt
        print("Pressione qualquer tecla para continuar...")
        msvcrt.getch()
    except ImportError:
        # Linux / macOS
        print("Pressione Enter para continuar...")
        input()