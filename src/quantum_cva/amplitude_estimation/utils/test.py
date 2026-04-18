

import re 

s = "hiçfadjs,ut=ESS"
match = re.search(r",ut=(\w{3})", s)
if match:
    result = match.group(1)
    if result=="ESS":
        print("yes")