# pip install xmltodict
#   CentOS: `$ sudo yum install python-xmltodict`
#   Ubuntu: `$ sudo apt install python-xmltodict`
import xmltodict


def parser(text, encoding=None):
    return xmltodict.parse(text, encoding=encoding)


def parser_file(file, encoding=None):
    with open(file, encoding=encoding) as temp:
        text = temp.read()
    return parser(text, encoding)


def parser_labelme(file, encoding=None):
    xml = parser_file(file, encoding)
    xml = xml["annotation"]["object"]

    if not isinstance(xml, list):
        xml = [xml]

    res = []
    for obj in xml:
        if "polygon" in obj:
            pts = []
            for pt in obj["polygon"]["pt"]:
                pts.append(int(pt["x"]))
                pts.append(int(pt["y"]))
            tmp = {"name": obj["name"],
                   "deleted": obj["deleted"],
                   "verified": obj["verified"],
                   "occluded": obj["occluded"],
                   "attributes": obj["attributes"],
                   "pts": pts}
            res.append(tmp)

    return res