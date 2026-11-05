from pymavlink import mavutil
import time
import pymap3d.enu as enu
import math

import sys

import threading

import utils.utils as utils 
import drone_control as drone_control

import json
from pathlib import Path
from typing import Tuple, Union

def ler_alvo_centro_json(caminho_json: Union[str, Path]) -> Tuple[float, float]:
    """
    Lê as coordenadas alvo_centro (lat, lon) de um arquivo JSON.

    Formato esperado:
    {
      "alvo_centro": { "lat": -22.8636306, "lon": -43.2215238 }
    }

    Retorna:
        (lat, lon) como tuple[float, float]

    Lança:
        FileNotFoundError, ValueError, KeyError, json.JSONDecodeError
    """
    caminho = Path(caminho_json)
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho.resolve()}")

    data = json.loads(caminho.read_text(encoding="utf-8"))

    alvo = data["alvo_centro"]
    lat = float(alvo["lat"])
    lon = float(alvo["lon"])

    # validações básicas
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude inválida: {lat}")
    
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude inválida: {lon}")

    return (lat, lon)

def configurar_parametros(drone, altitude):

    
    # Modificar parametro do drone para apontar para o home
    # drone_control.set_param(drone, "WP_YAW_BEHAVIOR", 1)  # 1 = Face next WP incluindo HOME no RTL

    # time.sleep(0.1)

    altitude_cm = int(altitude * 100.0)
    drone_control.set_param(drone, "RTL_ALT", altitude_cm)  # 1 = Face next WP incluindo HOME no RTL

def determina_ponto_falso(alvo_centro, ponto_falso_esquerda, ponto_falso_direita):

    distancia_ate_ponto_falso_direita = utils.distance_between_geo_points(alvo_centro, ponto_falso_direita)
    distancia_ate_ponto_falso_esquerda = utils.distance_between_geo_points(alvo_centro, ponto_falso_esquerda)

    lado_alvo_direita = None

    # utiliza o ponto falso mais distante do alvo
    if(distancia_ate_ponto_falso_direita > distancia_ate_ponto_falso_esquerda):
        ponto_falso = ponto_falso_direita
        lado_alvo_direita = False
        print("Ponto falso na DIREITA")

    else:
        ponto_falso = ponto_falso_esquerda
        lado_alvo_direita = True
        print("Ponto falso na ESQUERDA")


    return ponto_falso, lado_alvo_direita
    

def main():

    altitude_base = 6

    altitude_delta = 3

    raio  = 5.0

    # Coordenada do alvo
    try:
        alvo_centro = ler_alvo_centro_json("alvo_cluster.json")
    
    except Exception as e:
        print(f"[ERRO] Não foi possível ler alvo_centro do JSON: {e}")
        return

    print(f"Localização do alvo: {alvo_centro}")


    pontos_meio_campo = [
        (-22.9454166, -43.1567189),  # arquibancada
        (-22.9454000, -43.1566311),  # centro do campo
        (-22.9453802, -43.1565432)   # praia
    ]

    # pontos falsos candidados
    ponto_falso_direita = (-22.9450387, -43.1565191)
    ponto_falso_esquerda = (-22.9451215, -43.1569000)

    ponto_falso_escolhido, lado_alvo_direita = determina_ponto_falso(alvo_centro, ponto_falso_esquerda, ponto_falso_direita)

    ##################################################################
    # DRONE DE ATAQUE (DRONE 3) --- ESQUERDA
    port1 = "COM7"
    baud1 = 57600
    drone3_ataque = drone_control.connect_drone(port1, baud1)
    print("Conectado ao drone 1")

    # DRONE FINTA (DRONE 1 (FPV)) --- CENTRO
    port2 = "COM5" 
    baud2 = 57600
    drone1 = drone_control.connect_drone(port2, baud2)
    print("Conectado ao drone 2")

    # DRONE FINTA (DRONE 2) --- DIREITA
    port3 = "COM3"
    baud3 = 57600
    drone2 = drone_control.connect_drone(port3, baud3)
    print("Conectado ao drone 3")

    ##################################################################
    # Utilização com o simulador SITL
    
    # drone3_ataque = drone_control.connect_drone_sitl(host="127.0.0.1", port=5762)
    # print("Conectado ao drone 1")

    # drone1 = drone_control.connect_drone_sitl(host="127.0.0.1", port=5772)
    # print("Conectado ao drone 2")

    # drone2 = drone_control.connect_drone_sitl(host="127.0.0.1", port=5782)
    # print("Conectado ao drone 3")

    ##################################################################

    drones = []
    drones.append(drone3_ataque)
    drones.append(drone1)
    drones.append(drone2)

    if len(drones) != len(pontos_meio_campo):
        print("[ERRO]: Tamanho da lista DRONES é diferente da lista PONTOS_MEIO_CAMPO")
        exit()

    
    altitudes = [altitude_base + i * altitude_delta for i in range(len(drones))]
    print(f"Altitudes: {altitudes}")


    for drone in drones:
        drone_control.set_flight_mode(drone, "GUIDED")


    utils.delay(5)

    for drone in drones:
        msg = drone.recv_match(type='STATUSTEXT', blocking=True, timeout=2)
        if msg:
            print(f"STATUSTEXT: {msg.text}")

    

    print("Cadastrando recebimento de telemetria de POSIÇÃO...")
    for drone in drones:

        tentativas = 3

        for i in range(tentativas):

            # Telemetria GLOBAL_POSITION_INT
            drone.mav.command_long_send(
                drone.target_system, drone.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                33,      # ID da mensagem GLOBAL_POSITION_INT
                1000000 / 15,  # intervalo em microssegundos → 15 Hz
                0, 0, 0, 0, 0)
            
            time.sleep(0.2)


    print("Configurando parametros dos drones...")

    drone_control.set_velocidade_drones(drones, 400)

    # config_threads = []

    # for drone, altitude  in zip(drones, altitudes):

    #     th = threading.Thread(target=configurar_parametros, args=(drone, altitude))
    #     config_threads.append(th)
    #     th.start()

    # for th in config_threads:
    #     th.join()



    ############################# DECOLAGEM ##########################
    arm_threads = []

    for drone, altitude  in zip(drones, altitudes):

        tolerancia_m: float = 0.5
        timeout_s: float = 40.0

        th = threading.Thread(target=drone_control.arm_takeoff_and_wait, args=(drone, altitude, tolerancia_m, timeout_s))

        arm_threads.append(th)
    
    for th in arm_threads:
        th.start()
    
    for th in arm_threads:
        th.join()
    ####################################################################


    ############################# FILTRAGEM DOS DRONES ARMADOS ##########################
    # IMPORTANTE
    # Filtra apenas os drones que estão armados, mantendo o vínculo (drone, altitude)
    pares_armados = [(d, a, m) for d, a, m in zip(drones, altitudes, pontos_meio_campo) if d.motors_armed()]

    drones    = [d for d, _, _ in pares_armados]
    altitudes = [a for _, a, _ in pares_armados]
    pontos_meio_campo = [m for _, _, m in pares_armados]
    ###########################################################################

    goto_threads = []
    erro_aproximacao_waypoint = 2


    ############################  IDA AO MEIO CAMPO  ###############################
    for drone, ponto, altitude  in zip(drones, pontos_meio_campo, altitudes):
        th = threading.Thread(target=drone_control.goto_location_and_wait, args=(drone, ponto, altitude, erro_aproximacao_waypoint))
        goto_threads.append(th)
    
    for th in goto_threads:
        th.start()
    
    for th in goto_threads:
        th.join()
    ###########################################################################



    ############################  FINTA  ###############################
    numero_drones_finta = len(drones) - 1
    pontos_falsos = utils.generate_circle_points_enu(ponto_falso_escolhido, raio, numero_drones_finta, h=0)

    # 3 drones avançam para o alvo, mas apenas o drone de ataque (drone3_ataque) ataca o alvo, os outros dois fazem um "falso ataque" para confundir o inimigo
    for drone, ponto_falso, altitude in zip(drones[1:], pontos_falsos, altitudes[1:] ):
        threading.Thread(target=drone_control.goto_location_and_wait, args=(drone, ponto_falso, altitude, erro_aproximacao_waypoint)).start()
    ##################################################################

    if lado_alvo_direita:
        print("Delay ataque na direita do campo")
        utils.delay(5)

    ############################  ATAQUE DO ALVO  #####################
    drone_control.goto_location_and_wait(drones[0], alvo_centro, altitudes[0], erro_aproximacao_waypoint)
    
    drone_control.lancar_carga(drones[0])
    ##################################################################


    ######################## CONFIGURA RTL ##########################
    config_threads = []

    for drone, altitude  in zip(drones, altitudes):

        th = threading.Thread(target=configurar_parametros, args=(drone, altitude))
        config_threads.append(th)
        th.start()

    for th in config_threads:
        th.join()

    ########################################################################


    ####################  RETORNO DE TODOS OS DRONES  #####################
    rtl_threads = []

    print("Retornando com os drones...")

    for drone  in drones:

        th = threading.Thread(target=drone_control.set_flight_mode, args=(drone, "RTL"))
        rtl_threads.append(th)
    
    rtl_threads[0].start()

    if lado_alvo_direita:
        print("Delay RTL - ataque na direita do campo")
        utils.delay(10)

    for th in rtl_threads[1:]:
        th.start()
    
    for th in rtl_threads:
        th.join()
    ##################################################################

    print("Missão concluída.")

if __name__ == "__main__":
    main()