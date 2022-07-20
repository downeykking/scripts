from logger import setup_logger

# directory name -> output
logger = setup_logger(output="logs", name="test")
a = [1, 2, 3, 4, 5]
logger.info("a list {}".format(a))
