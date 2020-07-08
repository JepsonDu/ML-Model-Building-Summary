#### Remove tags happening at the end and print results

```
movie_tag = movie.rstrip("<\i>")
print(movie_tag)
```



#### Split the string using commas and print results
```
movie_no_comma = movie_tag.split(",")
print(movie_no_comma)
```

#### Join back together and print results
```
movie_join = " ".join(movie_no_comma)
print(movie_join)
```
#### Split string at line boundaries
```
file_split = file.split("\n")

for substring in file_split:
    substring_split = substring.split(",")
    print(substring_split)
```
#### Find, count and replace 
```
for movie in movies:
  	# Find if actor occurrs between 37 and 41 inclusive
    if movie.find("actor", 37, 42) == -1:
        print("Word not found")
    # Count occurrences and replace two by one
    elif movie.count("actor") == 2:  
        print(movie.replace("actor actor", "actor"))
    else:
        # Replace three occurrences by one
        print(movie.replace("actor actor actor", "actor"))
       
```   	
#### Assign the substrings to the variables
```
first_pos = wikipedia_article[3:19].lower()
second_pos = wikipedia_article[21:44].lower()

my_list.append("The tool {} is used in {}")

my_list.append("The tool {1} is used in {0}")

for my_string in my_list:
  	print(my_string.format(first_pos, second_pos))
```
- format the string by using format

```
# Create a dictionary
plan = {
  		"field": courses[0],
        "tool": courses[1]
        }

# Complete the placeholders accessing elements of field and tool keys
my_message = "If you are interested in {data[field]}, you can take the course related to {data[tool]}"

# Use dictionary to replace placeholders
print(my_message.format(data = plan))
```  	
format the datetime

```
# Import datetime 
from datetime import datetime

# Assign date to get_date
get_date = datetime.now()

# Add named placeholders with format specifiers
message = "Good morning. Today is {today:%B %d, %Y}. It's {today:%H:%M} ... time to work!"

# Format date
print(message.format(today=get_date))
```
f string to format the string, directly refill the placeholders

- !s (string version)
- !r (string containing a printable representation, i.e. with quotes)
- !a (some as !r but escape the non-ASCII characters)
- ：.2f remain 2 digital in the float

```
print(f"{field3} create around {fact3:.2f}% of the data but only {fact4:.1f}% is analyzed")

print(f"Data science is considered {field1!r} in the {fact1:d}st century")

```
### Substitute Template

- need to import Template package
- only can be used for user input 

```
# Import Template
from string import Template

# Create a template
wikipedia = Template("$tool is a $description")

#Substitute variables in template
print(wikipedia.substitute(tool=tool1, description=description1))

```

### Formate special format

```
from string import Template

# Select variables
our_tool = tools[0]
our_fee = tools[1]
our_pay = tools[2]

# Create template
course = Template("We are offering a 3-month beginner course on $tool just for $$ $fee ${pay}ly")

# Substitute identifiers with three variables
print(course.substitute(tool=our_tool, fee=our_fee, pay=our_pay))

```
### Regex r"regex"

- \d: one digit
- \w: one word character
- \W: non-word character
- \s: one whitespace
- \S in the regex to match anything but a whitespace

- Always start a regex with r. Remember that normal characters match themselves. Use \d to indicate digits and \W to indicate any non-word character, for example, & or #.

- To find all matches of pattern in a string, use the method .findall() from the re module. Don't forget to specify the pattern and the string as arguments.

- f"regex"  do not forget add f

```
# Import the re module
import re

# Write the regex
regex = r"@robot\d\W"

# Find all matches of regex
print(re.findall(regex, sentiment_analysis))

print(re.findall(r"number\sof\sretweets:\s\d", sentiment_analysis))
```

#### replace the regex
```
# Write a regex to match pattern separating sentences
regex_sentence = r"\W\dbreak\W"

# Replace the regex_sentence with a space
sentiment_sub = re.sub(regex_sentence, " ", sentiment_analysis)

# Write a regex to match pattern separating words
regex_words = r"\Wnew\w"

# Replace the regex_words and print the result
sentiment_final = re.sub(regex_words, ' ', sentiment_sub)
print(sentiment_final)

```
### repetition
- \d{3} find the string with5 digit eg 13245
- {n,m} n times at least, m times at most
- quantifier:	only apply to the charactors on the left
	- + once or more
	- * zero or more
	- ? zero or one
	
```
# Import re module
import re

for tweet in sentiment_analysis:
  	# Write regex to match http links and print out result
	print(re.findall(r"http\S+", tweet))

	# Write regex to match user mentions and print out result
	print(re.findall(r"@\w+", tweet))
```
#### Example 
Complete the for-loop with a regex that finds all dates in a format similar to 1st september 2019 17:25

```
# Complete the for loop with a regex to find dates
for date in sentiment_analysis:
	print(re.findall(r"\d{1,2}\w+\s\w+\s\d{4}\s\d{1,2}:\d{2}", date))
```
### replace the regex and split the string based on regex
- re.sub
- re.split

```
# Write a regex matching the hashtag pattern
regex = r"#\w+"

# Replace the regex by an empty string
no_hashtag = re.sub(regex, "", sentiment_analysis)

# Get tokens by splitting text
print(re.split(r"\s+", no_hashtag))

```

- Match any character (except newline): .
- Start ofthe string: ^   (only locate the regex at the start of the string)
- End ofthe string: $ (only locate the regex at the end of the string)
- Escape special characters: \  eg:\. to indentify the mark . note the matching note
- Or operator
	- Character | 
	- Set of characters: [ ]
	- exponential mark transfer the expression into negative
- [a-zA-Z] contain all the alphabet in the []
- [0-9] contain all the number in the []

####Example
The first part can contain:
Upper A-Z and lowercase letters a-z
Numbers
Characters: !, #, %, &, *, $, .
Must have @
Domain:
Can contain any word characters
But only .com ending is allowed

```
regex = r"[a-zA-Z!#%&*.0-9]+@\w+\.com"

```

#### match and search 
- match
- search

- greedy match  .+
- lazy match .+?

#### Example: 

1. Write a lazy regex expression to find all the (.......)

```
# Write a lazy regex expression to find all the (.......)
sentences_found_lazy = re.findall(r"\(.*?\)", sentiment_analysis)

```
####Grouping
1. extract the ID before @ of the email adress
	- using () to group a specific subset of a regex

```
# Write a regex that matches email
regex_email = r"([A-Za-z0-9]+)@\S+"

for tweet in sentiment_analysis:
    # Find all matches of regex in each tweet
    email_matched = re.findall(regex_email, tweet)

    # Complete the format method to print the results
    print("Lists of users found in this tweet: {}".format(email_matched))
``` 
#### Example

- (A|B){3} 乘法分配律
-  (:? regex) outgroup the ()   

Extract the movie name from the 'I love the movie Avengers'

```
# Write a regex that matches sentences with the optional words
regex_positive = r"(?:love|like|enjoy).+?(?:movie|concert)\s(.+?)\."

for tweet in sentiment_analysis:
	# Find all matches of regex in tweet
    positive_matches = re.findall(regex_positive, tweet)
    
    # Complete format to print out the results
    print("Positive comments found {}".format(positive_matches))
    

```

####extract the group from the groups = re.search
- groups.group(0) extract all groups
- groups.group(1) extract the first groups
- name the group: (?P \<group name> regex)
- group.group("group name") to extract the group

## lookaround
Positive lookahead (?=) makes sure that first part of the expression is followed by the lookahead expression. Positive lookbehind (?<=) returns all matches that are preceded by the specified pattern.

####example 
Get all the words that are preceded by the word python or Python in sentiment_analysis. Print out the words found.



```
# Positive lookbehind
look_behind = re.findall(r"(?<=[Pp]ython\s)\w+", sentiment_analysis)

# Print out
print(look_behind)
```
#### example 
Get all the words that are followed by the word python in sentiment_analysis. Print out the word found.


```
# Positive lookahead
look_ahead = re.findall(r"\w+(?=\spython)", sentiment_analysis)

# Print out
print(look_ahead)

```
