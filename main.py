import cv2

def main():
    videos_count = 0
    while videos_count < 3 :
        #videos
        if videos_count == 0:
            cap = cv2.VideoCapture("horses.mp4")
        if videos_count == 1:
            cap = cv2.VideoCapture("video_m_escuro.mp4")
        if videos_count == 2:
            cap = cv2.VideoCapture("carro.mp4")
        

        #resolução video
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # Lê o primeiro frame para referência inicial.
        ret, prev_frame = cap.read()
        if not ret:
            print("Não foi possível ler o frame inicial do vídeo.")
            return

        # redimenciona pra resolução 360p.
        prev_frame = cv2.resize(prev_frame, (640, 360))

        #converte escala de cinza
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            #le proximo frame
            ret, frame = cap.read()
            if not ret:
                break

            #Redimensiona para 360p.
            frame = cv2.resize(frame, (640, 360))

            # Converte o frame atual para escala de cinza.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #calcula diferença entre o frame atual e o anterior.
            diff = cv2.absdiff(prev_gray, gray)

            # desfoqueia a imagem de diferença para suavizar.
            diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)

            #threshold adaptativo para detectar movimento.
            motion_mask = cv2.adaptiveThreshold(
                diff_blur,              # imagem de entrada (desfoque da diferença entre frames)
                255,                    # valor máximo para o pixel, se a condição for verificada
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # método de cálculo do limiar utilizando média ponderada gaussiana
                cv2.THRESH_BINARY,      # tipo de threshold (binário)
                3,                      # tamanho do bloco (janela) para cálculo do limiar
                1                       # constante subtraída da média ponderada
            )

            #atualiza frame anterior.
            prev_gray = gray

            cv2.imshow("Frame Original (360p)", frame)
            cv2.imshow("Diferenca entre Frames (360p)", diff)
            cv2.imshow("Movimento (Threshold Adaptativo)", motion_mask)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        videos_count += 1

if __name__ == "__main__":
    main()
