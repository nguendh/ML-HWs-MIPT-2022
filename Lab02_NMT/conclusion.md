# Conclusion
## LAB02_ Neural_Machine_Translation
We are trying to cover understanding and implementing sequence-to-sequence (seq2seq) models by 3 models:
- 1st: The sample one: old deprecated code
- 2nd: Neural machine translation by jointly learning to align and translate
- 3rd: ✨Improve above model architecture by adding packed padded sequences and masking✨

## Blue Scores:

- 1st: 
```sh
9.76062377009099
```
- 2nd:
```sh
27.977001973843556
```
- 3rd:
```sh
27.742047347213866
```
##### So we can see that the simplest one takes lowest blue score. While the 2nd and the 3rd ones are implemented with added class `Attention`, take really high `Blue Score`. e.g `Blue Score` > 27
##### The 3rd one is an advanced version of the 2nd one, but they got the `Blue Score` approximately. In running time, the advanced one is really faster! So far till now, the quote `Attention Is All You Need` has been proven to be more than half complete :smiley:
