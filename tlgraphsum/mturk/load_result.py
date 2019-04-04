from tlgraphsum.mturk.post_questions import MTurkClient


def main():
    client = MTurkClient()
    client.batch_approve_hits()


if __name__ == "__main__":
    main()
