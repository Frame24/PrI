from transformers import AutoTokenizer, T5ForConditionalGeneration

model_name = "IlyaGusev/rut5_base_sum_gazeta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

article_text = "Впрочем, иногда странам удаётся вести конфликты более мирным путём. Например, Канада и Дания не могут поделить маленький остров Ханс, который вы можете видеть на иллюстрации. Поэтому на острове ведётся Hans Island так называемая «интеллигентная война». Раз в несколько месяцев туда прибывают военно‑морские силы Канады, устанавливают на острове флаг своего государства, поглощают заранее оставленный противником на острове запас крепких напитков, отмечают взятие острова и с победой отбывают."

input_ids = tokenizer(
    [article_text],
    max_length=600,
    add_special_tokens=True,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    no_repeat_ngram_size=4
)[0]

summary = tokenizer.decode(output_ids, skip_special_tokens=True)
print(summary)