# RL Final Project - Student ID 40400356

پروژه پایانی درس یادگیری تقویتی: طراحی، مدل‌سازی و تحلیل عامل هوشمند در هزارتوی پویا.

## خلاصه

این مخزن یک هزارتوی تصادفی-پویا را به‌صورت MDP پیاده‌سازی می‌کند و سه روش **Value Iteration**، **Q-Learning** و **SARSA(lambda)** را بدون استفاده از پیاده‌سازی‌های آماده RL مقایسه می‌کند. انتقال یادگیری فقط برای Q-Learning در دو محیط مقصد انجام شده است. رابط گرافیکی، داده خام، مدل‌های ذخیره‌شده، نمودارها، تست‌ها و گزارش نهایی نیز داخل مخزن قرار دارند.

## مشخصات اختصاصی پروژه

- شماره دانشجویی: `40400356`
- رقم یکی‌مانده‌به‌آخر و Seed پایه: `5`
- اندازه نقشه: `15 + (5 mod 4) = 16`، بنابراین `16 x 16`
- قابلیت تکمیلی: **دروازه دوره‌ای**
- فازهای باز دروازه: `0` و `1` از دوره چهارمرحله‌ای
- نمایش حالت:

```text
state = (row, column, has_key, gate_phase)
```

وجود `gate_phase` ضروری است؛ در غیر این صورت احتمال عبور از دروازه با دانستن حالت فعلی و عمل قابل تعیین نیست و خاصیت مارکوف نقض می‌شود.

## مدل انتقال

برای هر عمل انتخابی:

- احتمال اجرای همان عمل: `0.8`
- احتمال انحراف به هر جهت عمود: `0.1`
- برخورد با دیوار، در بسته یا دروازه بسته: ماندن در همان مکان با پیشروی فاز زمانی

سقف گام اپیزود برابر سه برابر تعداد خانه‌های قابل عبور است. این سقف به‌صورت TimeLimit و با برچسب `truncated` اعمال می‌شود و جزو MDP ایستای حل‌شده توسط Value Iteration نیست.

## تابع پاداش

دو حالت اجرا می‌شود:

1. `sparse`: پاداش اصلی برای کلید و هدف، همراه هزینه حرکت و جریمه برخورد/خطر.
2. `shaping`: پاداش sparse به‌اضافه shaping پتانسیل‌محور:

```text
F(s, s') = eta * [gamma * Phi(s') - Phi(s)]
```

پتانسیل بر اساس فاصله باقی‌مانده تا کلید و سپس هدف تعریف شده است. مقدار `eta=0.2` پس از stress-test انتخاب شد؛ مقیاس بزرگ‌تر حلقه‌های بازدهی نامناسب ایجاد می‌کرد.

## نصب

Python 3.10 یا جدیدتر پیشنهاد می‌شود.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

`Tkinter` معمولاً همراه Python نصب است. در برخی توزیع‌های Linux باید بسته `python3-tk` جداگانه نصب شود.

## اجرای اصلی

### رابط گرافیکی

```bash
python main.py
# یا
python main.py --gui
```

کنترل‌های GUI شامل انتخاب الگوریتم/محیط/حالت آموزش یا ارزیابی، شروع، توقف، ادامه، بازنشانی، اجرای مجدد، سرعت انیمیشن و نمایش سیاست است.

### اجرای دمو در ترمینال

```bash
python main.py --demo --episodes 2
```

### تولید مجدد نقشه‌های قطعی

```bash
python main.py --generate-maps
```

### بازتولید کامل آزمایش‌ها

```bash
python experiments/run_experiments.py --profile submission
```

برای smoke test سریع:

```bash
python experiments/run_experiments.py --profile quick
```

تمام نمودارها مستقیماً از CSVهای `results/raw_data/` ساخته می‌شوند. حذف اجرای ناموفق یا انتخاب دستی بهترین Seed انجام نشده است.

## تست‌ها

```bash
pytest -q
```

تست‌ها موارد زیر را پوشش می‌دهند:

- Seed و اندازه اختصاصی نقشه
- اعتبار BFS مسیر شروع به کلید و کلید به هدف
- جمع احتمال‌های انتقال
- منطق کلید، در و دروازه دوره‌ای
- truncation سقف گام
- به‌روزرسانی دستی Q
- رفتار lambda=0
- همگرایی Value Iteration
- انتقال تعدیل‌شده و انتخابی

## ساختار مخزن

```text
RL_FinalProject_40400356/
├── environments/
│   ├── maze.py
│   ├── generator.py
│   └── maps/
├── agents/
│   ├── common.py
│   ├── value_iteration.py
│   ├── q_learning.py
│   └── sarsa_lambda.py
├── transfer/transfer_learning.py
├── gui/
│   ├── app.py
│   └── renderer.py
├── experiments/
│   ├── run_experiments.py
│   ├── analysis.py
│   └── configs/
├── results/
│   ├── raw_data/
│   ├── models/
│   ├── figures/
│   └── videos/
├── tests/
├── docs/AI_USAGE.md
├── report.pdf
├── requirements.txt
├── README.md
└── main.py
```

## فایل‌های کلیدی نتایج

- `value_iteration_gamma.csv`: حساسیت gamma و زمان همگرایی
- `q_learning_epsilon_schedules.csv`: مقایسه کاهش خطی و نمایی epsilon
- `reward_shaping_comparison.csv`: sparse در برابر shaping
- `sarsa_lambda_ablation.csv`: آزمایش lambdaهای 0، 0.3، 0.7 و 0.9
- `three_algorithm_comparison.csv`: زمان، حافظه، موفقیت، کیفیت مسیر و توافق سیاست
- `transfer_episode_data.csv`: داده اپیزودی تمام سناریوهای انتقال
- `transfer_summary.csv`: عملکرد اولیه، سرعت یادگیری و عملکرد نهایی
- `negative_transfer_example.json`: نمونه عددی انتقال منفی
- `q_learning_manual_update_trace.csv`: بازسازی یک Q-update واقعی
- `sarsa_lambda_trace_log.csv`: تغییرات delta و E در چند گام
- `required_event_examples.jsonl`: نمونه همه رویدادهای اجباری

## نتایج بازتولیدشده اجرای نهایی

اجرای `submission` در این نسخه با پنج Seed برای آزمایش‌های اصلی و سه Seed برای انتقال، در مجموع `385.22` ثانیه طول کشید. اعداد زیر میانگین Seedها هستند و از فایل‌های خام مخزن استخراج شده‌اند، نه از بهترین اجرا:

- **Value Iteration** با `gamma=0.95` در 354 تکرار و 0.0578 ثانیه همگرا شد؛ نرخ موفقیت ارزیابی 1.000 و میانگین طول مسیر 31.36 گام بود.
- در **Q-Learning**، برنامه نمایی epsilon با وجود موفقیت نهایی تقریباً مساوی، زودتر به نرخ موفقیت 80 درصد رسید: میانه 397 اپیزود در برابر 470 اپیزود برای کاهش خطی. ارزیابی نهایی Q-Learning نرخ موفقیت 0.9992 و میانگین 36.67 گام داشت.
- در **SARSA(lambda)**، مقدار `lambda=0.7` بهترین تعادل را ایجاد کرد: موفقیت نهایی آموزشی 0.927، میانه 665 اپیزود تا آستانه 80 درصد و میانگین 71.10 گام. مقدار `lambda=0.9` ناپایدار شد و موفقیت نهایی آن فقط 0.006 بود.
- میانگین توافق دقیق سیاست با مرجع Value Iteration برای Q-Learning برابر 53.8 درصد و برای SARSA(lambda) برابر 52.3 درصد بود. این معیار به tieهای نزدیک حساس است و همراه نقشه اختلاف و کیفیت rollout تفسیر شده است.
- در مقصد مشابه، آموزش از صفر در بودجه 800 اپیزود شکست خورد، ولی انتقال کامل و انتقال‌های تعدیل‌شده/انتخابی به موفقیت نهایی 1.0 رسیدند. در مقصد متفاوت، `beta=0.25` سریع‌ترین اصلاح را داشت و در میانه 184 اپیزود به موفقیت 80 درصد رسید، در برابر 247 اپیزود برای کپی کامل و 366 اپیزود برای آموزش از صفر.
- نمونه انتقال منفی واقعی با regret برابر `17.695803` در `negative_transfer_example.json` ثبت شده است.

نتیجه مهم این است که «محیط مشابه‌تر» الزاماً «آسان‌تر» نیست و ضرب Q در beta مثبت نیز در لحظه شروع argmax را تغییر نمی‌دهد؛ اثر beta پس از شروع آموزش و از طریق اندازه خطای TD ظاهر می‌شود. تحلیل کامل این دو نکته در گزارش آمده است.

## منابع مفهومی

- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., 2018.
- Bellman, R. *Dynamic Programming*, 1957.
- Watkins, C. J. C. H., & Dayan, P. “Q-learning,” 1992.
- Rummery, G. A., & Niranjan, M. “On-line Q-learning using connectionist systems,” 1994.
- Ng, A. Y., Harada, D., & Russell, S. “Policy invariance under reward transformations,” 1999.

## شفافیت ابزارهای کمکی

جزئیات پیشنهادهای دریافت‌شده، اصلاح‌های انجام‌شده و دو نمونه پیشنهاد ناسازگار در `docs/AI_USAGE.md` و جدول متناظر گزارش نهایی ثبت شده است. هیچ کتابخانه آماده‌ای مانند Stable-Baselines یا RLlib استفاده نشده است.
