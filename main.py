import argparse
import collections
import pathlib
import random
import sqlite3
import types
import typing
from dataclasses import dataclass, field, fields, is_dataclass, make_dataclass

import httpx
import trio
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


def remove_escaping(thing):
    if thing is None:
        return thing

    return (
        thing.replace("&#39;", "'")
        .replace("&nbsp", " ")
        .replace("&rsquo;", "'")  # this should maybe be the special quote?
        .replace("&quot;", '"')
        .replace("&amp;", "&")
    )


@dataclass
class Metadata:
    pk: str | tuple[str, ...] | None
    fks: dict[str, tuple[type, str]] = field(default_factory=dict)
    extract: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)


def convert_faculty(faculty, term):
    # TODO: does email_address/display_name vary between terms?
    # note: there's also faculty["term"]
    assert faculty["category"] is None
    # assert faculty["emailAddress"] is not None, faculty
    return Faculty(
        banner_id=int(faculty["bannerId"]),
        display_name=remove_escaping(faculty["displayName"]),
        email_address=remove_escaping(faculty["emailAddress"]),
        term=term,
    )


@dataclass(frozen=True)
class Faculty:
    metadata: typing.ClassVar = Metadata(("banner_id", "term"))

    banner_id: int
    display_name: str
    email_address: str | None  # should this be not-stored?
    term: str  # TODO: consider if term->email_address should be moved to another table


def convert_section_faculty(section_faculty, term):
    return SectionFaculty(
        primary_indicator=section_faculty["primaryIndicator"],
        faculty=convert_faculty(section_faculty, term),
    )


@dataclass(frozen=True)
class SectionFaculty:
    metadata: typing.ClassVar = Metadata(None)

    primary_indicator: bool
    faculty: Faculty


def convert_meeting(meeting):
    assert meeting["category"] == meeting["meetingTime"]["category"]

    return Meeting(
        building=meeting["meetingTime"]["building"],
        building_description=remove_escaping(
            meeting["meetingTime"]["buildingDescription"]
        ),
        campus=meeting["meetingTime"]["campus"],
        campus_description=remove_escaping(meeting["meetingTime"]["campusDescription"]),
        begin_time=meeting["meetingTime"]["beginTime"],
        end_time=meeting["meetingTime"]["endTime"],
        credit_hour_session=meeting["meetingTime"]["creditHourSession"],
        hours_week=meeting["meetingTime"]["hoursWeek"],
        room=meeting["meetingTime"]["room"],
        meeting_schedule_type=meeting["meetingTime"]["meetingScheduleType"],
        meeting_type=meeting["meetingTime"]["meetingType"],
        meeting_type_description=meeting["meetingTime"]["meetingTypeDescription"],
        category=meeting["category"],
        monday=meeting["meetingTime"]["monday"],
        tuesday=meeting["meetingTime"]["tuesday"],
        wednesday=meeting["meetingTime"]["wednesday"],
        thursday=meeting["meetingTime"]["thursday"],
        friday=meeting["meetingTime"]["friday"],
        saturday=meeting["meetingTime"]["saturday"],
        sunday=meeting["meetingTime"]["sunday"],
    )


@dataclass(frozen=True)
class Meeting:
    metadata: typing.ClassVar = Metadata(
        None,
        extract=[
            ("building", ("building_description",)),
            ("campus", ("campus_description",)),
            ("meeting_type", ("meeting_type_description",)),
        ],
    )

    building: str | None
    building_description: str | None
    campus: str | None
    campus_description: str | None
    begin_time: str | None
    end_time: str | None
    credit_hour_session: float | None
    hours_week: float | None
    room: str | None
    meeting_schedule_type: str
    meeting_type: str | None
    meeting_type_description: str | None
    category: str  # ??

    monday: bool
    tuesday: bool
    wednesday: bool
    thursday: bool
    friday: bool
    saturday: bool
    sunday: bool


def convert_section_attribute(section_attribute):
    assert not section_attribute["isZTCAttribute"]

    return SectionAttribute(
        code=section_attribute["code"],
        description=remove_escaping(section_attribute["description"]),
    )


@dataclass(frozen=True)
class SectionAttribute:
    metadata: typing.ClassVar = Metadata("code")

    code: str
    description: str


def convert_course(course, term):
    assert course["anySections"] is None
    assert course["attributes"] is None
    assert course["ceu"] in [None, True, False]  # ??
    assert course["courseLevels"] is None
    assert course["courseScheduleTypes"] is None
    assert course["description"] is None
    # assert course["division"] is None, course  # TODO: make sure all uses of this are trivial
    assert course["durationUnit"] is None
    assert course["gradeModes"] is None
    assert course["numberOfUnits"] is None
    assert course["preRequisiteCheckMethodCde"] == "B"
    assert course["subjectCode"] == course["subject"]

    # TODO: try to understand creditHourHigh vs creditHourLow
    # assert section["creditHourHigh"] is None
    # assert section["creditHourIndicator"] is None
    # assert section["creditHourLow"] is section["creditHours"]

    # ^ same for billHourHigh, billHourIndicator, and billHourLow
    # ^ same for labHourHigh, labHourIndicator, and labHourLow
    # ^ same for lectureHourHigh, lectureHourIndicator, and lectureHourLow
    # ^ same for otherHourHigh, otherHourIndicator, and otherHourLow

    # TODO: what is termEffective? it seems like garbage data.

    return Course(
        college=course["college"],
        college_code=course["collegeCode"],
        description=remove_escaping(course["courseDescription"]),
        number=course["courseNumber"],
        title=remove_escaping(course["courseTitle"]),
        department=course["department"],
        department_code=course["departmentCode"],
        subject=course["subject"],
        subject_description=remove_escaping(course["subjectDescription"]),
        term_start=course["termStart"],
        term_end=course["termEnd"],
        term=term,
    )


@dataclass
class Course:
    metadata: typing.ClassVar = Metadata(
        ("subject", "number", "term"),
        extract=[
            ("subject", ("subject_description",)),
            ("college", ("college_code",)),
            ("department_code", ("department",)),
        ],
    )

    college: str
    college_code: str
    description: str | None
    number: str
    title: str
    department: str | None
    department_code: str | None
    subject: str
    subject_description: str
    term_start: str
    term_end: str

    # `term` is necessary as the description can vary from term to term.
    # (and likely other things as well...)
    # (it may be worth sticking those into another class for database size reduction reasons)
    term: str


def convert_section(section):
    # TODO: try to understand creditHourHigh vs creditHourLow
    # assert section["creditHourHigh"] is None
    # assert section["creditHourIndicator"] is None
    # assert section["creditHourLow"] is section["creditHours"]

    # TODO: add crosslisting links
    # assert section["crossList"] is None or section["crossList"] in ["QL", "QN"]

    # TODO: what does this mean?
    # assert section["linkIdentifier"] is None or section["linkIdentifier"] == "A"

    assert section["reservedSeatSummary"] is None

    return Section(
        course_number=section["courseNumber"],
        subject=section["subject"],
        course_reference_number=int(section["courseReferenceNumber"]),
        course_title=remove_escaping(section["courseTitle"]),
        credit_hours=section["creditHours"],
        enrollment=section["enrollment"],
        faculty=[
            convert_section_faculty(f, section["term"]) for f in section["faculty"]
        ],
        maximum_enrollment=section["maximumEnrollment"],
        schedule_type_description=section["scheduleTypeDescription"],
        sequence_number=section["sequenceNumber"],
        wait_capacity=section["waitCapacity"],
        wait_count=section["waitCount"],
        meetings=[convert_meeting(m) for m in section["meetingsFaculty"]],
        instructional_method=section["instructionalMethod"],
        instructional_method_description=section["instructionalMethodDescription"],
        section_attributes=[
            convert_section_attribute(a) for a in section["sectionAttributes"]
        ],
        term=section["term"],
        term_description=section["termDesc"],
        crosslist_capacity=section["crossListCapacity"],
        crosslist_count=section["crossListCount"],
        campus_description=remove_escaping(section["campusDescription"]),
        open_section=section["openSection"],
        part_of_term=section["partOfTerm"],
        is_section_linked=section["isSectionLinked"],
    )


@dataclass
class Section:
    metadata: typing.ClassVar = Metadata(
        ("course_reference_number", "term"),
        fks={"subject, course_number, term": (Course, "subject, number, term")},
        extract=[
            ("instructional_method", ("instructional_method_description",)),
            ("term", ("term_description",)),
        ],
    )

    course_number: str
    course_reference_number: int
    course_title: str  # differs from course.title
    credit_hours: float | int | None
    enrollment: int
    faculty: list[SectionFaculty]
    maximum_enrollment: int
    schedule_type_description: str
    sequence_number: str
    subject: str
    wait_capacity: int
    wait_count: int
    meetings: list[Meeting]
    instructional_method: str | None
    instructional_method_description: str | None
    section_attributes: list[SectionAttribute]
    term: str
    term_description: str
    crosslist_capacity: int | None
    crosslist_count: int | None
    campus_description: str
    open_section: bool
    part_of_term: str  # ??
    is_section_linked: bool


@dataclass
class Linkage:
    metadata: typing.ClassVar = Metadata(
        None,
        fks={
            "course_reference_number, term": (Section, "course_reference_number, term")
        },
    )

    link_id: int
    course_reference_number: int
    term: str


def typecheck(value, vtype):
    if is_dataclass(vtype):
        assert isinstance(value, vtype)
        for field in fields(vtype):
            try:
                typecheck(getattr(value, field.name), field.type)
            except AssertionError as e:
                raise AssertionError(
                    f"{vtype.__name__}.{field.name} = {getattr(value, field.name)} does not satisfy {field.type}"
                ) from e

    elif isinstance(vtype, type):
        assert isinstance(value, vtype)

    elif typing.get_origin(vtype) is list:
        assert isinstance(value, list)
        for v in value:
            typecheck(v, typing.get_args(vtype)[0])

    elif typing.get_origin(vtype) is types.UnionType:
        for t in typing.get_args(vtype):
            try:
                typecheck(value, t)
            except AssertionError:
                continue
            else:
                break
        else:
            raise AssertionError(f"{value} does not satisfy {vtype}")

    else:
        raise AssertionError(f"Unknown type {vtype}")


sections = []
courses = []
linkages = []
section_links = set()
temp_section_links = []


async def fetch(client, *args, **kwargs):
    for _ in range(5):
        try:
            r = await client.get(*args, **kwargs)
        except (httpx.RemoteProtocolError, httpx.ReadError):
            await trio.sleep(4 * random.random())
            continue

        if r.status_code == 500:
            # IDK why these happen sporadically
            await trio.sleep(4 * random.random())
            continue

        break

    r.raise_for_status()
    return r


async def fetch_sections_page(client, term, offset):
    r = await fetch(
        client,
        "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/searchResults/searchResults",
        params={
            "txt_term": term,
            "pageOffset": offset,
            "pageMaxSize": "500",
            "sortColumn": "subjectDescription",
            "sortDirection": "asc",
        },
    )

    r.raise_for_status()
    json = r.json()
    assert json["success"]
    assert json["data"] is not None
    return json


async def fetch_courses_page(client, term, offset):
    r = await fetch(
        client,
        "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/courseSearchResults/courseSearchResults",
        params={
            "txt_term": term,
            "pageOffset": offset,
            "pageMaxSize": "500",
            "sortColumn": "subjectDescription",
            "sortDirection": "asc",
        },
    )

    r.raise_for_status()
    json = r.json()
    assert json["success"]
    assert json["data"] is not None
    return json


async def fetch_linked_sections(client, term, crn):
    r = await fetch(
        client,
        "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/searchResults/fetchLinkedSections",
        params={"term": term, "courseReferenceNumber": crn},
    )

    r.raise_for_status()
    json = r.json()
    return {
        tuple(sorted([crn] + [s["courseReferenceNumber"] for s in link]))
        for link in json["linkedData"]
    }


async def setup_cookies(client):
    r = await client.get(
        "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/classSearch/classSearch",
        follow_redirects=True,
    )
    r.raise_for_status()


async def get_terms(client):
    r = await client.get(
        "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/classSearch/getTerms?searchTerm=&offset=1&max=500"
    )
    r.raise_for_status()
    terms = {t["code"]: t["description"] for t in r.json()}
    assert len(terms) < 500
    return terms


async def enter_term(client, term):
    r = await client.post(
        "https://registration.banner.gatech.edu/StudentRegistrationSsb/ssb/term/search?mode=search",
        data={
            "term": term,
        },
    )
    r.raise_for_status()


async def get_sections(client, progress, term, description):
    start = 0

    task = progress.add_task(f"Reading sections for {description}", total=None)

    while True:
        json = await fetch_sections_page(client, term, start)
        for s in json["data"]:
            if s["isSectionLinked"]:
                temp_section_links.append(s["courseReferenceNumber"])
        sections.extend(convert_section(s) for s in json["data"])
        progress.update(task, total=json["totalCount"], advance=len(json["data"]))

        if start + len(json["data"]) >= json["totalCount"]:
            break

        start += len(json["data"])


async def get_courses(client, progress, term, description):
    start = 0

    task = progress.add_task(f"Reading courses for {description}", total=None)

    while True:
        json = await fetch_courses_page(client, term, start)
        courses.extend(convert_course(c, term) for c in json["data"])
        progress.update(task, total=json["totalCount"], advance=len(json["data"]))

        if start + len(json["data"]) >= json["totalCount"]:
            break

        start += len(json["data"])


async def get_linked_sections(client, progress, term, task, chan):
    global section_links

    async with chan:
        async for message in chan:
            try:
                section_links |= await fetch_linked_sections(client, term, message)
            except httpx.HTTPStatusError:
                # ultimately some things will fail
                # this can be detected by checking section's isSectionLinked
                # (though actually, that can miss some things. that's life!)
                continue
            finally:
                progress.update(task, advance=1)


async def get_data(term):
    async with httpx.AsyncClient(
        timeout=120,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; classes/1.0; +https://github.com/horo-fox/classes)"
        },
    ) as client:
        await setup_cookies(client)

        terms = await get_terms(client)
        if term is not None:
            if term not in terms:
                raise ValueError(f"invalid term {term}, expected " + ", ".join(terms))

            terms = {term: terms[term]}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            metatask = progress.add_task("[orange_red1]Downloading data")

            for term, description in progress.track(terms.items(), task_id=metatask):
                await enter_term(client, term)

                async with trio.open_nursery() as nursery:
                    nursery.start_soon(
                        get_sections, client, progress, term, description
                    )
                    nursery.start_soon(get_courses, client, progress, term, description)

                if temp_section_links:
                    tx, rx = trio.open_memory_channel(0)
                    task = progress.add_task(
                        f"Reading section links for {description}",
                        total=len(temp_section_links),
                    )
                    async with trio.open_nursery() as nursery:
                        for _ in range(20):
                            nursery.start_soon(
                                get_linked_sections,
                                client,
                                progress,
                                term,
                                task,
                                rx.clone(),
                            )
                        async with tx:
                            for link in temp_section_links:
                                await tx.send(link)
                    temp_section_links.clear()

                for task in progress.tasks:
                    if task.finished:
                        progress.update(task.id, visible=False)

                for i, link in enumerate(section_links):
                    for part in link:
                        linkages.append(
                            Linkage(
                                link_id=i, course_reference_number=int(part), term=term
                            )
                        )

                section_links.clear()


def init(args):
    trio.run(get_data, None)


def update(args):
    trio.run(get_data, args.term)


parser = argparse.ArgumentParser(prog="classes")
subparsers = parser.add_subparsers(required=True)

parser_init = subparsers.add_parser("init")
parser_init.set_defaults(func=init, type="init")

parser_update = subparsers.add_parser("update")
parser_update.add_argument("term")
parser_update.set_defaults(func=update, type="update")

args = parser.parse_args()
args.func(args)

console.print(
    f"saving [blue]{len(sections)} sections[/blue] and "
    f"[green]{len(courses)} courses[/green] to a database"
)
for section in sections:
    typecheck(section, Section)
for course in courses:
    typecheck(course, Course)
for linkage in linkages:
    typecheck(linkage, Linkage)

mapping = {
    str: "TEXT NOT NULL",
    int: "INT NOT NULL",
    float: "REAL NOT NULL",
    bool: "TEXT NOT NULL",
}


def get_pk(t, pk):
    assert pk is not None
    if isinstance(pk, tuple):
        return tuple(getattr(t, k) for k in pk)
    return (getattr(t, pk),)


def type_to_sql(t):
    if t in mapping:
        return mapping[t]
    if typing.get_origin(t) is types.UnionType:
        ts = list(typing.get_args(t))
        assert type(None) in ts
        ts.remove(type(None))
        if len(ts) == 2:
            assert t == float | int | None
            return type_to_sql(float).replace(" NOT NULL", "")
        assert len(ts) == 1
        return type_to_sql(ts[0]).replace(" NOT NULL", "")
    if typing.get_origin(t) is list:
        # many-to-many (potentially)
        onto = typing.get_args(t)
        assert len(onto) == 1
        onto = onto[0]
        assert is_dataclass(onto), onto

        assert thing.metadata.pk is not None
        if onto.metadata.pk is not None:
            to_add.append((onto, [a for b in vs for a in b]))
            tpk = thing.metadata.pk
            opk = onto.metadata.pk
            tpk = tpk if isinstance(tpk, tuple) else (tpk,)
            opk = opk if isinstance(opk, tuple) else (opk,)

            connector = make_dataclass(
                f"{thing.__name__}{onto.__name__}Connector",
                (
                    *[
                        (tk, next(f.type for f in fields(thing) if f.name == tk))
                        for tk in tpk
                    ],
                    *[
                        (ok, next(f.type for f in fields(onto) if f.name == ok))
                        for ok in opk
                    ],
                ),
            )

            sql_tpk = thing.metadata.pk
            sql_opk = onto.metadata.pk
            sql_tpk = ", ".join(sql_tpk) if isinstance(sql_tpk, tuple) else sql_tpk
            sql_opk = ", ".join(sql_opk) if isinstance(sql_opk, tuple) else sql_opk

            connector.metadata = Metadata(
                None, {sql_tpk: (thing, sql_tpk), sql_opk: (onto, sql_opk)}
            )
            to_add.append((
                connector,
                [
                    connector(
                        *get_pk(t, thing.metadata.pk), *get_pk(a, onto.metadata.pk)
                    )
                    for t, v in zip(things, vs)
                    for a in v
                ],
            ))
        else:
            # nvm must be a 1-to-many
            assert onto not in added
            assert onto not in to_add
            tpk = thing.metadata.pk
            tpk = tpk if isinstance(tpk, tuple) else (tpk,)

            new_onto = make_dataclass(
                onto.__name__,
                [
                    (
                        f"parent_{tk}",
                        next(f.type for f in fields(thing) if f.name == tk),
                    )
                    for tk in tpk
                ],
                bases=(onto,),
                frozen=True,
            )
            to_add.append((
                new_onto,
                [
                    new_onto(
                        **{
                            field.name: getattr(a, field.name) for field in fields(onto)
                        },
                        **{f"parent_{tk}": getattr(thing, tk) for tk in tpk},
                    )
                    for thing, b in zip(things, vs)
                    for a in b
                ],
            ))

            sql_tpk = thing.metadata.pk
            sql_from_tpk = (
                ", ".join([f"parent_{k}" for k in sql_tpk])
                if isinstance(sql_tpk, tuple)
                else f"parent_{sql_tpk}"
            )
            sql_tpk = ", ".join(sql_tpk) if isinstance(sql_tpk, tuple) else sql_tpk
            onto.metadata.fks[sql_from_tpk] = (thing, sql_tpk)
        return "SKIP"
    if is_dataclass(t):
        # 1-1 or 1-many
        assert t.metadata.pk is not None
        # (1-1 is not yet implemented)
        pk = (t.metadata.pk,) if isinstance(t.metadata.pk, str) else t.metadata.pk
        for k in pk:
            typ = type_to_sql(next(f.type for f in fields(t) if f.name == k))
            assert typ != "SKIP"

            columns.append(f"{k} {typ}")
            column_values.append([python_to_sql_value(getattr(v, k)) for v in vs])

        thing.metadata.fks[", ".join(pk)] = (t, ", ".join(pk))

        to_add.append((t, vs))

        return "SKIP"
    raise AssertionError(f"unknown type {t!r}")


def python_to_sql_value(val):
    # all this could *probably* just use prepared statements...
    if val is None:
        return "NULL"
    if isinstance(val, bool):
        return "'true'" if val else "'false'"
    if isinstance(val, tuple):
        raise AssertionError("unexpected tuple")
    if isinstance(val, list):
        raise AssertionError("unexpected list")
    if is_dataclass(val):
        raise AssertionError("unexpected dataclass")
    if isinstance(val, str):
        return "'" + val.replace("'", "''") + "'"
    return repr(val)


def pascal_to_snake(name):
    parts = []
    start = 0
    skipped_to = 0
    for i, c in enumerate(name):
        if c.isupper():
            if skipped_to == i:
                skipped_to += 1
            else:
                parts.append(name[start:i])
                start = i
                skipped_to = i + 1

    parts.append(name[start:])
    return "_".join(part.lower() for part in parts)


added = {}
to_add = collections.deque()
to_add.append((Section, sections))
to_add.append((Course, courses))
to_add.append((Linkage, linkages))
tables = []
inserts = []

while to_add:
    thing, things = to_add.popleft()
    assert is_dataclass(thing)

    # make sure pk is unique
    if thing.metadata.pk is not None:
        ts = []
        key_mapping = {}
        for t in things:
            k = get_pk(t, thing.metadata.pk)
            if k in key_mapping:
                assert key_mapping[k] == t, f"{key_mapping[k]} != {t}"
            else:
                key_mapping[k] = t
                ts.append(t)

        things = ts

    if thing in added:
        assert added[thing] == things
        continue

    added[thing] = things
    columns = []
    column_values = []
    skip_keys = []

    # handle extracted tables
    for extract_pk, extract_other_keys in thing.metadata.extract:
        assert isinstance(extract_other_keys, tuple)
        skip_keys.extend(extract_other_keys)

        extracted_class = make_dataclass(
            extract_pk.capitalize(),
            (
                (
                    extract_pk,
                    next(f.type for f in fields(thing) if f.name == extract_pk),
                ),
                *[
                    (ok, next(f.type for f in fields(thing) if f.name == ok))
                    for ok in extract_other_keys
                ],
            ),
        )
        extracted_class.metadata = Metadata(extract_pk)
        thing.metadata.fks[extract_pk] = (extracted_class, extract_pk)

        # ensure the transformation doesn't lose information
        for t in things:
            if getattr(t, extract_pk) is None:
                for v in get_pk(t, extract_other_keys):
                    assert v is None

        extracted_things = [
            extracted_class(getattr(t, extract_pk), *get_pk(t, extract_other_keys))
            for t in things
            if getattr(t, extract_pk) is not None  # pks cannot be None
        ]
        to_add.append((extracted_class, extracted_things))

    for f in fields(thing):
        vs = [getattr(v, f.name) for v in things]

        t = type_to_sql(f.type)
        if (
            t != "SKIP" or f.name in skip_keys
        ):  # type_to_sql queued the necessary followup already
            columns.append(f"{f.name} {t}")
            column_values.append(list(map(python_to_sql_value, vs)))

    for fk, to in thing.metadata.fks.items():
        columns.append(
            f"FOREIGN KEY({fk}) REFERENCES {pascal_to_snake(to[0].__name__)}({to[1]})"
            " ON DELETE CASCADE"
        )

    if thing.metadata.pk is not None:
        pk = (
            thing.metadata.pk
            if isinstance(thing.metadata.pk, tuple)
            else (thing.metadata.pk,)
        )
        columns.append(f"PRIMARY KEY({', '.join(pk)})")

    tables.append(
        f"CREATE TABLE {pascal_to_snake(thing.__name__)} ({', '.join(columns)}) STRICT;"
    )

    inserted_values = ",".join([f"({', '.join(vs)})" for vs in zip(*column_values)])
    if inserted_values:
        inserts.append(
            f"INSERT INTO {pascal_to_snake(thing.__name__)} VALUES {inserted_values}"
            + (" ON CONFLICT DO NOTHING;" if args.type == "update" else "")
        )

database_path = pathlib.Path.cwd() / "classes.db"

if args.type == "init":
    database_path.unlink(missing_ok=True)

con = sqlite3.connect(str(database_path), autocommit=False)
cur = con.cursor()
cur.execute("PRAGMA foreign_keys = ON;")

if args.type == "init":
    for t in tables:
        cur.execute(t)

else:
    # this should probably be dynamically generated...
    cur.execute(f"DELETE FROM course WHERE term = '{args.term}';")
    cur.execute(f"DELETE FROM faculty WHERE term = '{args.term}';")

con.commit()

for insert in inserts:
    cur.execute(insert)
con.commit()
