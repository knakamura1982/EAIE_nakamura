import cv2
import numpy as np


# 四角形検出器
class SquareDetector:

    @staticmethod
    def cos_angle(pt1, pt2, pt0) -> float:
        dx1 = float(pt1[0, 0] - pt0[0, 0])
        dy1 = float(pt1[0, 1] - pt0[0, 1])
        dx2 = float(pt2[0, 0] - pt0[0, 0])
        dy2 = float(pt2[0, 1] - pt0[0, 1])
        v = np.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2))
        return (dx1*dx2 + dy1*dy2) / v

    # コンストラクタ
    #   - n_lebels: 二値化の際の閾値を何段階に変化させるか
    #   - area_threshold: 検出対象とする四角形の面積の最小値
    #   - canny_threshold: Cannyフィルタ適用時の閾値
    def __init__(self, n_levels=16, area_threshold=10000, canny_threshold=50):
        self.n_levels = n_levels
        self.area_threshold = area_threshold
        self.canny_threshold = canny_threshold
        return

    # 検出
    #   - img: 入力画像
    #   - multi_squares: 複数の検出結果を返すか否か
    def detect(self, img:np.ndarray, multi_squares:bool=True) -> list:

        squares = [] # 出力変数の初期化

        # 入力画像を 1 チャンネル画像に変換する
        channels = 1 if len(img.shape) == 2 else img.shape[2]
        if channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # ノイズ低減のため，一度拡大したのち縮小
        H, W = img.shape[:2]
        img = cv2.pyrUp(cv2.pyrDown(img, dstsize=(W//2, H//2)), dstsize=(W, H))

        # 二値化の閾値を変化させながら複数回検出処理を試行
        largest_area = 0
        largest_square = []
        for l in range(0, self.n_levels):

            if l == 0:
                # 閾値 == 0 のときは Canny フィルタによるエッジ検出結果を使用
                gray = cv2.Canny(img, self.canny_threshold, 5)
                gray = cv2.dilate(gray, None) # 膨張処理を行い，短いエッジを連結
            else:
                # 二値化
                gray = np.zeros((H, W), dtype=np.uint8)
                gray[img >= l*255/self.n_levels] = 0
                gray[img < l*255/self.n_levels] = 255

            # 輪郭検出
            contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for i, cnt in enumerate(contours):

                # 輪郭を多角形で近似して角を取得
                arclen = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, arclen*0.02, True)

                # 面積の算出
                area = abs(cv2.contourArea(approx))

                # 凸形状で，角が4つあり，面積が一定以上に大きく，かつ各辺のなす角の cos の最大値が 0.3 未満であるもののみを残す
                if approx.shape[0] == 4 and area > self.area_threshold and cv2.isContourConvex(approx):
                    maxCosine = 0
                    break_flag = False
                    for j in range(4):
                        if approx[j][0, 0] == 0 or approx[j][0, 1] == 0 or approx[j][0, 0] == W-1 or approx[j][0, 1] == H-1:
                            break_flag = True
                    if break_flag:
                        continue
                    for j in range(2, 5):
                        cosine = abs(SquareDetector.cos_angle(approx[j%4], approx[j-2], approx[j-1]))
                        maxCosine = max(maxCosine, cosine)
                    if maxCosine < 0.3:
                        if area > largest_area:
                            largest_area = area
                            largest_square = [approx]
                        squares.append(approx)

        return squares if multi_squares else largest_square

    # 検出枠の描画
    #   - img: 描画先の画像
    #   - squares: self.detect()メソッドによる検出結果（四角形のリスト）
    #   - color: 描画する矩形の色（B, G, Rの順）
    #   - thickness: 描画する矩形の枠の太さ
    def draw(self, img:np.ndarray, squares:list, color:tuple=(64, 255, 64), thickness:int=3) -> None:
        cv2.polylines(img, squares, True, color, thickness=thickness, lineType=cv2.LINE_8)
        return
