# Plugins

## [katex](https://github.com/lzanini/mdbook-katex)

```plaintext
\[
\text{MSE}(q) = \mathbb{E}_X \left[ d(q(x), x)^2 \right] = \int p(x) d(q(x), x)^2 dx,
\]
```

\[
\text{MSE}(q) = \mathbb{E}_X \left[ d(q(x), x)^2 \right] = \int p(x) d(q(x), x)^2 dx,
\]

## [admonish](https://github.com/tommilligan/mdbook-admonish)

    ```admonish success

    ```

```admonish success

```

## [reading-time](https://github.com/pawurb/mdbook-reading-time)

```plaintext
全文字数: **{{ #没有空格word_count }}**

阅读时间: **{{ #没有空格reading_time }}**
```

全文字数: **{{ #word_count }}**

阅读时间: **{{ #reading_time }}**

## [alert](https://github.com/lambdalisue/rs-mdbook-alerts)


    > [!CAUTION]
    > Negative potential consequences of an action.


> [!NOTE]  
> Highlights information that users should take into account, even when skimming.

> [!TIP]
> Optional information to help a user be more successful.

> [!IMPORTANT]  
> Crucial information necessary for users to succeed.

> [!WARNING]  
> Critical content demanding immediate user attention due to potential risks.

> [!CAUTION]
> Negative potential consequences of an action.
>

## [repl](https://github.com/MR-Addict/mdbook-repl)

这不是 jupyter notebook，所以请把代码写在一个 codeblock 中。（你可以编辑 codeblock ）

```python
class Greeter:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return "Hello, " + self.name

g = Greeter("world")
print(g.greet())
```



## [mermaid](https://github.com/badboy/mdbook-mermaid.git)

```plaintext
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

```mermaid
graph TD;
    A-->B;
    A-->C;
    B-->D;
    C-->D;
```

## [embedify](https://github.com/MR-Addict/mdbook-embedify.git)

```text
{%这是一个空格embed bilibili id="BV1uT4y1P7CX" loading="lazy" %}
```

{% embed bilibili id="BV1uT4y1P7CX" loading="lazy" %}
