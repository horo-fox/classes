Browse Georgia Tech's course listings as a database.

### Using the database

You can download an initial database from <https://classes.horo.services/classes.db>, or use the script in `init` mode. Then you can update the data for any given term in `update` mode.

If the database is at `classes.db`, it is an SQLite database and you can use any tools that work on SQLite databases on it. For instance, starting in Python 3.12 this will spin up a terminal for queries:

```
$ python -m sqlite3 classes.db
```

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
