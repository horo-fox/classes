Browse Georgia Tech's course listings as a database.

### Set up

Use a virtual environment, please!

On Windows:

```
$ python -m venv .venv
$ .venv/Scripts/activate
```

On Unix:

```
$ python -m venv .venv
$ . .venv/bin/activate
```

Now install the requirements:

```
# make sure you are in the virtual environment
$ pip install -r requirements.txt
```

### Using the script

You can either initialize a database with:

```
# make sure you are in the virtual environment
$ python main.py init
```

Or you can update a database for a given term with:

```
# make sure you are in a virtual environment
$ python main.py update TERM
```

For instance, use `202508` for the Fall 2025 term. Generally terms are `<year><monthcode>` where the month is `02` for Spring, `05` for Summer, and `08` for Fall.

### Development

Please use the following commands:
 - `ruff format`
 - `ruff check`
 - `uv pip compile requirements.in --universal -o requirements.txt`
