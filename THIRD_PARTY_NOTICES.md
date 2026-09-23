# Third-party notices

## forecasting-tools (MIT License)

This bot uses [forecasting-tools](https://github.com/Metaculus/forecasting-tools) as a
dependency. Two parts of this repository are adapted from it:

- `metabot/cdf.py` — `standardize_cdf` mirrors `NumericDistribution._standardize_cdf`
  (Metaculus CDF constraints), with a slightly larger uniform floor.
- `metabot/bot.py` — conditional-question handling follows `template_bot_2026_fall.py`.

```
MIT License

Copyright (c) 2024 CodexVeritas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## metac-bot-template

The official [Metaculus bot template](https://github.com/Metaculus/metac-bot-template)
has no licence file. No file from it is copied here; this bot follows its documented
structure (ForecastBot subclass, GitHub Actions schedule, secret names).
