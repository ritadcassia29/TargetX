# TargetX
# Sistema de Controle e Gerenciamento de Drones

## Sobre o projeto

Este repositório reúne o desenvolvimento de um sistema de controle e gerenciamento de drones, desenvolvido em Python, com foco na integração entre controle de voo, visão computacional e execução de missões autônomas.

O projeto explora a comunicação com os veículos aéreos, o processamento de informações visuais e a coordenação das diferentes etapas de uma missão, integrando ferramentas de visão computacional e comunicação MAVLink.

## Objetivos

- Desenvolver módulos de controle e comunicação com drones.
- Implementar mecanismos de busca e identificação visual de alvos.

## Organização dos arquivos

O repositório está organizado em três módulos principais:

### 1. Drone Control
Responsável pelas funções de controle e comunicação com os drones, utilizando a biblioteca `pymavlink` e o protocolo MAVLink.

Contempla a comunicação com o piloto automático, o gerenciamento dos comandos de voo e a movimentação dos veículos.

### 2. Busca do Alvo
Módulo dedicado ao processamento visual e à identificação de alvos por meio de técnicas de visão computacional e detecção de objetos com YOLO.

O sistema utiliza as informações obtidas pelas imagens para auxiliar no processo de localização e acompanhamento visual.

### 3. Ataque do Drone
Módulo que reúne a lógica de uma etapa específica da missão, integrando as informações de controle e percepção desenvolvidas nos demais componentes.

## Considerações finais

Este repositório documenta o desenvolvimento dos componentes de software do projeto, reunindo os códigos produzidos e a integração entre diferentes ferramentas de programação, controle de sistemas embarcados e visão computacional.

O trabalho proporciona a aplicação prática de conhecimentos de programação, automação, processamento de imagens e sistemas autônomos.
