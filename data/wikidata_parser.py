import numpy as np
import os
import bz2
import random
import json
import qwikidata
from qwikidata.json_dump import WikidataJsonDump
from qwikidata.linked_data_interface import get_entity_dict_from_api
from collections import Counter
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme
import sqlite3
from itertools import islice
import time
from pprint import pprint
import traceback

TOTAL_SIZE = 93396717 -2 #based on decompressing and calling both "wc -l" and "grep -c $", two of the lines are "[" and "]"
RANDOM_SAMPLE = .2
DATAFILE = 'latest-all.json.bz2'

class DatabaseExtractor():
    def __init__(self, dbname='wikidata_claims_refs_parsed.db'):
        self.dbname = dbname
        self.prepare_extraction()
        
    def finish_extraction(self):
        self.db.commit()
        
    def prepare_extraction(self):
        self.db = sqlite3.connect(self.dbname)
        self.cursor = self.db.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims(
                entity_id TEXT,
                claim_id TEXT,
                claim_rank TEXT,
                property_id TEXT,
                datatype TEXT,
                datavalue TEXT,
                PRIMARY KEY (
                    claim_id
                )
        )''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS claims_refs(
                claim_id TEXT,
                reference_id TEXT,
                PRIMARY KEY (
                    claim_id,
                    reference_id
                )
        )''')

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS refs(
                reference_id TEXT,
                reference_property_id TEXT,
                reference_index TEXT,
                reference_datatype TEXT,
                reference_value TEXT,
                PRIMARY KEY (
                    reference_id,
                    reference_property_id,
                    reference_index
                )
        )''')
        self.db.commit()  
        
    def extract_claim(self, entity_id, claim):
        if claim['mainsnak']['snaktype'] == 'value':
            value = str(claim['mainsnak']['datavalue'])
        else:
            value = claim['mainsnak']['snaktype']
        try:
            self.cursor.execute('''
            INSERT INTO claims(entity_id, claim_id, claim_rank, property_id, datatype, datavalue)
            VALUES($var,$var,$var,$var,$var,$var)'''.replace('$var','?'), (
                entity_id,claim['id'],claim['rank'],
                claim['mainsnak']['property'],claim['mainsnak']['datatype'],value
            ))
        except UnicodeEncodeError:
            print(entity_id,claim['id'],claim['rank'],
                claim['mainsnak']['property'],claim['mainsnak']['datatype'],value)
            raise
        except sqlite3.IntegrityError as err:
            #self.db.rollback()
            self.cursor.execute(
                '''SELECT *
                FROM claims 
                WHERE claim_id=$var
                '''.replace('$var','?'), (claim['id'],)
            )
            conflicted_value = self.cursor.fetchone()
            if conflicted_value == (entity_id,claim['id'],claim['rank'],
                    claim['mainsnak']['property'],claim['mainsnak']['datatype'],value):
                pass
            else:
                print(err, claim['id'])
                traceback.print_exc()
                raise err
        finally:
            #self.db.commit()
            pass

    def extract_reference(self, ref):
        for snaks in ref['snaks'].values():
            for i, snak in enumerate(snaks):
                if snak['snaktype'] == 'value':
                    value = str(snak['datavalue'])
                else:
                    value = snak['snaktype']
                try:
                    self.cursor.execute('''
                    INSERT INTO refs(reference_id, reference_property_id, reference_index,
                    reference_datatype, reference_value)
                    VALUES($var,$var,$var,$var,$var)'''.replace('$var','?'), (
                        ref['hash'],snak['property'],str(i),snak['datatype'],value
                    ))
                except sqlite3.IntegrityError as err:
                    #self.db.rollback()
                    self.cursor.execute(# WE DONT USE THE INDEX HERE, THEY TEND TO COME SHUFFLED FROM API AND SORTING TAKES TOO LONG
                        '''SELECT reference_id, reference_property_id, reference_datatype, reference_value
                        FROM refs 
                        WHERE reference_id = $var
                        AND reference_property_id = $var
                        '''.replace('$var','?'), (ref['hash'],snak['property'])
                    )
                    conflicted_values = self.cursor.fetchall()
                    if  (ref['hash'],snak['property'],snak['datatype'],value) in conflicted_values:
                        pass
                    else:
                        print(err, ref['hash'],snak['property'],i)
                        print('in the db:', conflicted_value)
                        print('trying to insert:',(ref['hash'],snak['property'],str(i),snak['datatype'],value))
                        traceback.print_exc()
                        raise err
                finally:
                    #self.db.commit()
                    pass
            
    def extract_claim_reference(self, claim, ref):
        claim['id'],ref['hash']
        try:
            self.cursor.execute('''
            INSERT INTO claims_refs(claim_id, reference_id)
            VALUES($var,$var)'''.replace('$var','?'), (
                claim['id'],ref['hash']
            ))
        except sqlite3.IntegrityError as err:
            #db.rollback()
            pass
        finally:
            #self.db.commit()
            pass
    
    def extract_entity(self, e):
        for outgoing_property_id in e['claims'].values():
            for claim in outgoing_property_id:
                self.extract_claim(e['id'],claim)
                if 'references' in claim:
                    for ref in claim['references']: 
                        self.extract_claim_reference(claim, ref)
                        self.extract_reference(ref)

def consume(iterator, n=None):
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        collections.deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)
        
def parse_picks(picks, last_pick, f, savepoints, extractor, get_sizes=False, verbose=True):
    sizes = []
    extracted_picks = []
    '''
    - picks is the list of picked positions in the dump to extract
    - last_pick: is the last picked position in the dump (from picks) to have been succesfully extracted to the DB.
    when it is -1 it means no picked position from the dump was extracted. This is returned at the end to mark where
    the extraction stopped.
    - f is the dump file itself
    - savepoints is the actual byte position in the file where the entity in the respective picked position was found. For
    example, savepoints[42] returns the byte position in the file where the entity in position picks[24] of the dump begins.
    - index_start is the index on the picks list where to start reading next. If last_pick is -1, then index_start
    should be 0, as the first entity should be the one to be extracted next. Otherwise, it should begin on the entity
    immediately after last_pick, which means picks.index(last_pick)+1.
    - extraction_fun is what to do with each entity
    ''' 
    if last_pick == -1:
        index_start = 0
        f.seek(0) #If last_pick is -1, no extraction took place and we seek the beginning of the file
        consume(f,1) #skipping first [
    else:
        '''
        If the extraction is continuing from a past run, then last_pick is the picked position last extracted successfully.
        Thus we need to jump to the position the file reader would be if it didn't finish halfway through.
        This would be on the position right after consuming the entity represented by last_pick.
        To find that, we do:
        '''
        index_start = picks.index(last_pick)+1 # As described at the start, we take the index of the pick following last_pick
        # to start the loop
        f.seek(savepoints[index_start-1]) # We get the position in the file where the last_pick entity starts 
        consume(f,1)    # and then we skip that entity
    try:
        for i, pick in enumerate(picks[index_start:]):
            '''
            The first line here is consume(f, pick - last_pick - 1). This is because:
            If the extraction is starting from zero, last_pick is -1 and we consume pick - (-1) -1, which is pick, so
            we jump 'pick' number of lines and land on the 'pick'st line, as pick is 0-indexed 
            Exemple: pick is 0, so we jump 0 lines and get the first element of the dump.
            Exemple: pick is 42, so we jump 42 lines and get the 43rd element of the dump; which, picks being 0-indexed,
            is what pick=42 means!
            
            If the extraction is continuing a loop that started from scratch, then pick is the picked position following
            last_pick in the picks list. In this case, we jump the lines between the end of last_pick and the beginning of
            pick. This would be pick- last_pick -1, and not just pick - last_pick, as we are skipping the number of entities 
            BETWEEN them.
            Exemple: last_pick is 42, and pick is 46. We already read entity 42, so the reader is at the very start of 
            43. We jump 43, 44, and 45, landing on the beginning of 46, so 3 entities. 46-42-1 = 3.
            
            If, however, the extraction is following another extraction after successfully extracting a last_pick entity,
            then as per the code above, pick will start by being the picked position following last_pick in the picks list,
            so we go back to the case above.
            '''
            consume(f, pick - last_pick - 1)
            savepoints[index_start+i] = f.tell() # We register the starting position of this entity in the savepoints
            linebytes = f.readline() # We consume the entity. 
            # THIS HAS PUT THE POINTER AT THE END OF THIS ENTITY/START OF THE NEXT
            if get_sizes:
                sizes.append(len(linebytes))
            line_str = linebytes.decode('utf-8') # We decode the bytes
            line_str = line_str.rstrip(',\n')
            entity = json.loads(line_str)
            #print('{} : {} : {} : {}'.format(index_start+i,pick,entity['id'], savepoints[index_start+i]))
            if verbose:
                print(str(pick/TOTAL_SIZE*100)+'%'+20*' ',end='\r')
            extractor.extract_entity(entity)        
            last_pick = pick
            extracted_picks.append(pick)
    except Exception as err:
        print(err, entity['id'])
        traceback.print_exc()
        raise err
    finally:
        extractor.finish_extraction()
        with open('extracted_picks.txt','w+') as f:
            for extracted_pick in extracted_picks:
                f.write(str(extracted_pick))
                f.write('\n')
        if get_sizes:
            return last_pick, savepoints, sizes
        return last_pick, savepoints
        
def reset_picks(filename, picks):
    f = bz2.open(filename, mode="rb")
    last_pick = -1
    savepoints = [None] * len(picks)
    return last_pick, f, savepoints


def sanity_check(picks):
    # SANITY CHECK TO SEE IF THE EXTRACTION IS GETTING THE PICKS, CHECK THE FIRST Nth PICKS
    wjd = WikidataJsonDump(DATAFILE)
    first_n = []
    n = 100
    for i,e in enumerate(wjd):
        if i in picks[:n]:
            first_n.append(e['id'])
        if len(first_n) == n:
            break

    first_n_extractions = []
    class TestExtractor():
        def __init__(self, extraction_list):
            self.extraction_list = extraction_list
        def extract_entity(self, e):
            self.extraction_list.append(e['id'])
        def finish_extraction(self):
            pass
    extractor = TestExtractor(first_n_extractions)

    last_pick, f, savepoints = reset_picks(DATAFILE, picks)
    last_pick, savepoints = parse_picks(picks[:n], last_pick, f, savepoints, extractor, verbose=False)

    assert first_n == extractor.extraction_list, "Did not pass sanity check"

def main():
    # GENERATE THE PICKED POSITIONS REPRESENTING 20% OF THE DUMPS   
    print('Sampling picks ...')
    if RANDOM_SAMPLE < 1:
        np.random.seed(42) #we use the estimated size as seed too because why not
        picks = np.random.choice(TOTAL_SIZE,size=int(RANDOM_SAMPLE*TOTAL_SIZE),replace=False) # We randomly pick 20% of total
        picks = sorted(picks)
    else:
        picks = list(range(TOTAL_SIZE))
    print('Sampled {} picks:'.format(len(picks)),picks[:20],'...')
        
    print('Sanity checking ...')   
    sanity_check(picks)
        
    print('Setting up database ...')
    extractor = DatabaseExtractor()
    
    print('Start parsing ...')
    last_pick, f, savepoints = reset_picks(DATAFILE, picks)
    last_pick, savepoints = parse_picks(picks, last_pick, f, savepoints, extractor)
    print('Last pick was:',last_pick)
       
if __name__ == '__main__':
    main()
