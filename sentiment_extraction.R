# https://towardsdatascience.com/understanding-and-writing-your-first-text-mining-script-with-r-c74a7efbe30f 
# https://medium.com/swlh/exploring-sentiment-analysis-a6b53b026131
# https://cran.r-project.org/web/packages/syuzhet/vignettes/syuzhet-vignette.html

setwd("C:/Users/m/Documents/Python/PythonScripts/intro2ml_2021_final_project/Data")

# LIBRARIES ---------------------------------------------------------------

library(dplyr)
library(syuzhet)
library(tm)
library(wordcloud)

# IMPORT RAW DATA ---------------------------------------------------------
dataset <- utils::read.csv("dataset_2k.csv", header = TRUE, sep = ",")

# TEXT MINING -------------------------------------------------------
# myStopwords <- c(generics::setdiff(tm::stopwords('english'), c("r", "big")),"use")

dictCorpus <- dataset$tweet %>%
              tm::VectorSource() %>% 
              tm::Corpus() %>% 
              tm::tm_map(tm::content_transformer(tolower)) %>% # to lower case
              tm::tm_map(tm::content_transformer(gsub), #remove emojis
                         pattern = "\\W", replace = " ") %>% 
              tm::tm_map(tm::content_transformer( # remove URLs
                         function(x) gsub("http[^[:space:]]*", "", x))) %>% 
              tm::tm_map(tm::content_transformer( # remove anything other Eng letters/spaces
                         function(x) gsub("[^[:alpha:][:space:]]*", "", x))) %>% 
              tm::tm_map(tm::removeWords, tm::stopwords("english")) %>% # remove stopwords
              #tm::tm_map(tm::removeWords, myStopwords) %>% # remove custom stop words
              tm::tm_map(tm::stripWhitespace) %>% # remove extra spaces
              tm::tm_map(tm::removeNumbers) %>% # remove numbers
              tm::tm_map(tm::removePunctuation) # remove punctuation

myCorpus <- dictCorpus %>% # stemming
            tm::tm_map(tm::stemDocument) %>%
            tm::tm_map(tm::stemCompletion, dictionary = dictCorpus) %>% 
            utils::stack() %>% tibble::as_tibble() %>% dplyr::select(-values)

dataset <- dataset %>% dplyr::mutate(tweet = as.character(myCorpus$ind))

# SENTIMENT EXTRACTION -----------------------------------------------------------
output <- data.frame(anger = numeric(),
                     anticipation = numeric(),
                     disgust = numeric(),
                     fear = numeric(),
                     joy = numeric(),
                     sadness = numeric(),
                     surprise = numeric(),
                     trust = numeric(),
                     negative = numeric(),
                     positive = numeric())

for (row in 1:nrow(dataset)) {
  df <- dataset[row, "tweet"] %>% 
        #syuzhet::get_sentences() %>% # sentiment by sentence
        syuzhet::get_tokens(pattern = "\\W") %>% # sentiment by word
        syuzhet::get_nrc_sentiment(lang = "english", lowercase = TRUE) %>%
        colSums() %>% 
        t() %>% 
        data.frame()  
  output <- rbind(output, df)
}


# OUTPUT ------------------------------------------------------------------
utils::write.csv(output, file = "sentiment.csv")

# plot a word cloud 
wordcloud::wordcloud(dictCorpus,
                     scale = c(5, 0.5),    # Set min and max scale
                     max.words = 100,      # Set top n words
                     random.order = FALSE, # Words in decreasing freq
                     rot.per = 0.35,       # % of vertical words
                     use.r.layout = FALSE, # Use C++ collision detection
                     colors = brewer.pal(8, "Dark2"))


#lsf.str("package:dplyr") #to list all functions in package
#environmentName(environment(select)) #to get package name from function