## CANLI FUTBOL MAÇLARI İÇİN KENDİ VERİMİZLE SKOR TAHMİN MODELLEMESİ KURMAK



 ![pred](images/pred.jpg)



Tamamen kişisel merak üzerine çıkan fakat aslında pek orijinal olmayan bir fikir üzerine böyle bir proje yapmak istedim. Hali hazırda skor tahmin sitelerini görmüşsünüzdür. Neredeyse tamamı belirli bir ücret karşılığı hizmet veren bu sitelerden ziyade kendim tahmin üretecek bir model geliştirmek istedim.

Yaptığım bu çalışmanın birçok eksik yanı var farkındayım. Aslında bu eksiklikleri isteyerek bıraktım. İlk versiyon olarak düşündüğüm bu projeyi hem Türkçe kaynak olarak kullanılabilmesi için hem de bir sonraki aşama için ön hazırlık olmasını istedim. Bir sonraki versiyonda eksikliklerin bir bölümü tamamlanacak ve "deploy" edilmeye tamamen hazır gelecek. Son versiyonda ise Flask kullanılarak deployment işlemi yapılacak. Diğer versiyonları İngilizce olarak paylaşacağım ve buradaki gibi detaylı anlatım yapmak yerine sadece kısa notlar ekleyeceğim.

Bu versiyonda;

- Öznitelik mühendisliği yapılmamıştır.
- Maç dakikası değişkeni kullanılmamıştır.
- Kodlarda düzeltme işlemleri ayrıntılı olarak yapılmamıştır.
- Toplam skor tahmini için sadece light gbm regressor kullanılmıştır.
- Hiperparametre optimizasyonu detaylı olarak yapılmamıştır.
- Toplam skor tahmini ile ilgili karar mekanizması oluşturulmamıştır.

Şu an için aklıma gelen eksiklikler bunlar. Diğer versiyonlarda bu eksiklikler ve aklıma şu an için gelmeyen eksiklikler tamamlanacaktır.

Şimdi gelelim projeyi olabildiğince detaylı olarak anlatmaya

Kullanılan kütüphaneler aşağıda gördüğünüz gibidir. Eğer sizde yüklü olmayan kütüphane var ise *pip install kütüphane_ismi* şeklinde yükleyebilirsiniz. **Requests** ve **BeautifulSoup** kütüphanelerini veriyi çekerken , **lightgbm** kütüphanesini model oluşturmak için kullanacağız.



### Data Information



![scr](images/scr.png)



**League:**  Oynanan maçın hangi lige ait olduğunu gösterir.

**Minutes:** Oynanan maçın kaçıncı dakikada olduğunu gösterir.

**Home:** Ev sahibi takımın ismini gösterir.

**Score:** Maçın o dakikadaki skorunu gösterir.

**Away:** Deplasman takımının ismini gösterir.

**Corner:** Parantez içinde ilk yarıdaki korner sayılarını parantez dışında ise ev sahibi ve deplasman takımı için o ana kadar toplam kullanılmış korner sayısını gösterir.

**Dangerous Attack:** Turuncu olarak belirtilen ilk yarıdaki tehlikeli atak sayısını, yeşil olan gösterilen ise o ana kadar gelişen tehlikeli atak sayısını gösterir.

**Shots:** Turuncu olarak gösterilen ilk yarıdaki şut sayısını, yeşil olan ise o ana kadar gelişen tehlikeli atak sayısını gösterir.



Açıklamada belirtildiği üzere bu kodlar ilk versiyona aittir. Bir sonraki versiyonda geliştirmeler uygulanacaktır.



