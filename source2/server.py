# coding: utf-8
import os
import tornado.ioloop
import tornado.web

import predictor


class MainHandler(tornado.web.RequestHandler):
    """
    トップページを返す
    """
    def get(self):
        self.render("index.html")


class RecognitionHandler(tornado.web.RequestHandler):
    """
    画像を受け取り、識別結果を返す
    """
    def post(self):
        image_path = ""

        # 画像データと拡張子を変数に格納
        files = self.request.files
        image = files['image'][0]['body']
        mime = files['image'][0]['content_type']
            
        # jpeg画像の場合
        if mime == "image/jpeg":
            image_path = "images/image.jpg"
                
        # png画像の場合
        elif mime == "image/png":
            image_path = "images/image.png"
                
        # サポートされていない形式だった場合
        else:
            self.redirect('/')

        # 一時ファイルに画像を書き込み
        f = open(image_path, "wb")
        f.write(image)
        f.close()

        # 画像の識別
        top5 = predictor.predict_image(image_path)
        probs = []
        for i, data in enumerate(top5):
            probs.append("{0}. {2} / 確率：{1:.5}%".format(i+1, data[0]*100, data[1]))
                
        self.render("recognition.html", image_path=image_path, probs=probs)

                

application = tornado.web.Application(
    [
        (r"/", MainHandler),
        (r"/recognize", RecognitionHandler)
    ],
    template_path=os.path.join(os.getcwd(), "templates"),
    static_path=os.path.join(os.getcwd(), "static")
)


if __name__ == "__main__":
    application.listen(8000)
    tornado.ioloop.IOLoop.instance().start()

    
                                      
