import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.global.opencv_dnn.*;
import org.bytedeco.opencv.global.opencv_imgproc.*;
import org.bytedeco.opencv.global.opencv_highgui.*;

public class PoseDetection {
    public static void main(String[] args) {
        VideoCapture cap = new VideoCapture("assets/sample.mp4");
        cap.set(Videoio.CAP_PROP_FPS, 1);
        cap.set(Videoio.CAP_PROP_FRAME_WIDTH, 800);
        cap.set(Videoio.CAP_PROP_FRAME_HEIGHT, 800);

        Net net = Dnn.readNetFromTensorflow("assets/graph_opt.pb");

        int inWidth = 368;
        int inHeight = 368;
        float thr = 0.2f;

        while (true) {
            Mat frame = new Mat();
            cap.read(frame);

            if (frame.empty()) {
                break;
            }

            int frameWidth = frame.cols();
            int frameHeight = frame.rows();

            Mat inputBlob = Dnn.blobFromImage(frame, 1.0, new Size(inWidth, inHeight), new Scalar(127.5, 127.5, 127.5), true, false);

            net.setInput(inputBlob);

            Mat output = net.forward();
            output = output.reshape(1, 1, 19, 3);

            assert (19 == output.size(1));

            Point[] points = new Point[19];
            for (int i = 0; i < 19; i++) {
                Mat heatMap = new Mat();
                output.col(i).row(0).convertTo(heatMap, CvType.CV_32F);
                Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(heatMap);

                double x = (frameWidth * minMaxLocResult.maxLoc.x) / output.size(3);
                double y = (frameHeight * minMaxLocResult.maxLoc.y) / output.size(2);

                points[i] = (minMaxLocResult.maxVal > thr) ? new Point(x, y) : null;
            }

            for (String[] pair : POSE_PAIRS) {
                String partFrom = pair[0];
                String partTo = pair[1];

                int idFrom = BODY_PARTS.get(partFrom);
                int idTo = BODY_PARTS.get(partTo);

                if (points[idFrom] != null && points[idTo] != null) {
                    Imgproc.line(frame, points[idFrom], points[idTo], new Scalar(0, 255, 0), 3);
                    Imgproc.ellipse(frame, points[idFrom], new Size(3, 3), 0, 0, 360, new Scalar(0, 0, 255), Core.FILLED);
                    Imgproc.ellipse(frame, points[idTo], new Size(3, 3), 0, 0, 360, new Scalar(0, 0, 255), Core.FILLED);
                }
            }

            double t = net.getPerfProfile()[0] / Core.getTickFrequency();
            Imgproc.putText(frame, String.format("%.2fms", t), new Point(10, 20), Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 0));

            HighGui.imshow("Human Body", frame);

            if (HighGui.waitKey(1) == 'q') {
                break;
            }
        }

        cap.release();
        HighGui.destroyAllWindows();
    }

    private static final String[][] POSE_PAIRS = {
            {"Neck", "RShoulder"}, {"Neck", "LShoulder"}, {"RShoulder", "RElbow"},
            {"RElbow", "RWrist"}, {"LShoulder", "LElbow"}, {"LElbow", "LWrist"},
            {"Neck", "RHip"}, {"RHip", "RKnee"}, {"RKnee", "RAnkle"}, {"Neck", "LHip"},
            {"LHip", "LKnee"}, {"LKnee", "LAnkle"}, {"Neck", "Nose"}, {"Nose", "REye"},
            {"REye", "REar"}, {"Nose", "LEye"}, {"LEye", "LEar"}
    };

    private static final java.util.Map<String, Integer> BODY_PARTS = Map.of(
            "Nose", 0, "Neck", 1, "RShoulder", 2, "RElbow", 3, "RWrist", 4,
            "LShoulder", 5, "LElbow", 6, "LWrist", 7, "RHip", 8, "RKnee", 9,
            "RAnkle", 10, "LHip", 11, "LKnee", 12, "LAnkle", 13, "REye", 14,
            "LEye", 15, "REar", 16, "LEar", 17, "Background", 18
    );
}
