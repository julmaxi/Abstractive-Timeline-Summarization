import boto3


SANDBOX_ENDPOINT = "https://mturk-requester-sandbox.us-east-1.amazonaws.com"


class GrammaticalityTestMTurkClient:
    QUESTION = """

<HTMLQuestion xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
  <HTMLContent><![CDATA[
    <!DOCTYPE html>
      <body>
        <script src="https://assets.crowd.aws/crowd-html-elements.js"></script>
<!-- You must include this JavaScript file -->
<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>

<!-- For the full list of available Crowd HTML Elements and their input/output documentation,
      please refer to https://docs.aws.amazon.com/sagemaker/latest/dg/sms-ui-template-reference.html -->

<!-- You must include crowd-form so that your task submits answers to MTurk -->
<crowd-form answer-format="flatten-objects">

    <!-- The crowd-classifier element will create a tool for the Worker to select the
           correct answer to your question -->
    <crowd-classifier 
      name="quality"
      categories="['1 (Very bad)', '2', '3', '4', '5 (Very good)']"
      header="How readable is the provided sentence?"
    >

      <classification-target>
        {text}
      </classification-target>

     <!-- Use the short-instructions section for quick instructions that the Worker
            will see while working on the task. Including some basic examples of 
            good and bad answers here can help get good results. You can include 
            any HTML here. -->
      <short-instructions>
       Rate the readability of the text
      </short-instructions>

      <!-- Use the full-instructions section for more detailed instructions that the 
            Worker can open while working on the task. Including more detailed 
            instructions and additional examples of good and bad answers here can
            help get good results. You can include any HTML here. -->
      <full-instructions header="Sentiment Analysis Instructions">
          
        <p><strong>Positive</strong> sentiment include: joy, excitement, delight</p>
        <p><strong>Negative</strong> sentiment include: anger, sarcasm, anxiety</p>
        <p><strong>Neutral</strong>: neither positive or negative, such as stating a fact</p>
        <p><strong>N/A</strong>: when the text cannot be understood</p>
        <p>When the sentiment is mixed, such as both joy and sadness, use your judgment to choose the stronger emotion.</p>
      </full-instructions>

    </crowd-classifier>
</crowd-form>     
      </body>
    </html>
  ]]></HTMLContent>
  <FrameHeight>0</FrameHeight>
</HTMLQuestion>
"""

    def __init__(self, sandbox=True):
        args = {}
        if sandbox:
            args["endpoint_url"] = SANDBOX_ENDPOINT
        self.client = boto3.client('mturk', **args)

        self._hittype = None

    def create_hittype(self):
        response = self.client.create_hit_type(
            AutoApprovalDelayInSeconds=360,
            AssignmentDurationInSeconds=60,
            Reward='0.01',
            Title='Readability',
            Keywords='nlp',
            Description='Test',
            QualificationRequirements=[]
        )

        return response["HITTypeId"]

    @property
    def hittype(self):
        if not self._hittype:
            self._hittype = self.create_hittype()
        return self._hittype

    def upload_batch(self, values):
        for value in values:
            self.client.create_hit_with_hit_type(
                HITTypeId=self.hittype,
                MaxAssignments=3,
                LifetimeInSeconds=360 * 78,
                Question=self.QUESTION.format(text=value["text"]),
                RequesterAnnotation=value["identifier"])


if __name__ == "__main__":
    example_values = [{"text": "This is a test.", "identifier": "1"}, {"text": "This is a test.", "identifier": "1"}, {"text": "This is a test.", "identifier": "1"}, {"text": "This is a test.", "identifier": "1"}, {"text": "This is a test.", "identifier": "1"}]

    client = GrammaticalityTestMTurkClient()
    client.upload_batch(example_values)
