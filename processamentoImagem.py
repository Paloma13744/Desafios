import cv2
import numpy as np
from matplotlib import pyplot as plt

def contar_elementos(imagem):
    branco = np.array([255, 255, 255])  # Estrelas
    vermelho = np.array([255, 0, 0])    # Meteoros
    azul = np.array([0, 0, 255])        # Água
    
    # Máscaras para cada elemento
    mascara_estrelas = cv2.inRange(imagem, branco, branco)
    mascara_meteoros = cv2.inRange(imagem, vermelho, vermelho)
    mascara_agua = cv2.inRange(imagem, azul, azul)
    
    # Contando estrelas e meteoros
    num_estrelas, _ = cv2.connectedComponents(mascara_estrelas)
    num_meteoros, _ = cv2.connectedComponents(mascara_meteoros)
    
    # Contagem de meteoros que cairão na água
    num_meteoros_na_agua = 0
    
    # Verificação de meteoros caindo sobre a água
    for x in range(mascara_meteoros.shape[1]):
        coluna_meteoros = mascara_meteoros[:, x]
        coluna_agua = mascara_agua[:, x]
        
        # Se existir um meteoro acima de qualquer ponto de água, considera-se que ele cairá na água
        if np.any(coluna_meteoros) and np.any(coluna_agua):
            pos_meteoro = np.argmax(coluna_meteoros)  
            pos_agua = np.argmax(coluna_agua)         
            if pos_meteoro < pos_agua:                
                num_meteoros_na_agua += 1
    
 
    contornos, _ = cv2.findContours(mascara_estrelas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Desenhar os contornos das estrelas na imagem original
    imagem_contornada = imagem.copy()
    cv2.drawContours(imagem_contornada, contornos, -1, (255, 255, 255), 2) 

    # Adicionando o texto com a quantidade de estrelas na imagem
    texto = f"Numero de estrelas: {num_estrelas - 1}"  # Usar "Numero" sem acento
    cv2.putText(imagem_contornada, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return num_estrelas - 1, num_meteoros - 1, num_meteoros_na_agua, imagem_contornada

def main(caminho_imagem):
   
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        print("Erro: Imagem não encontrada ou não pode ser carregada.")
        return
    
    # Contagem dos elementos e obtenção da imagem com contornos
    num_estrelas, num_meteoros, num_meteoros_na_agua, imagem_contornada = contar_elementos(imagem)
    
    print(f"Número de estrelas: {num_estrelas}")
    print(f"Número de meteoros: {num_meteoros}")
    print(f"Meteoros que cairão na água: {num_meteoros_na_agua}")

    # Exibe a imagem original e a imagem com contornos
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    plt.title("Imagem Original")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(imagem_contornada, cv2.COLOR_BGR2RGB))
    plt.title("Estrelas Contornadas")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main('meteor_challenge_01.png') 