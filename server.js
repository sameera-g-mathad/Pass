const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const nodemailer = require("nodemailer");
const moment = require("moment");
dotenv.config({ path: "./config.env" });
const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static("./serve"));

app.route("/").post(async (req, res) => {
  try {
    const { toEmail, message } = req.query;
    const transporter = nodemailer.createTransport({
      service: process.env.SERVICE,
      port: process.env.EPORT,
      auth: {
        user: process.env.USER,
        pass: process.env.PASSWORD,
      },
    });
    const mailoptions = {
      from: process.env.SERVER_MAIL,
      to: toEmail,
      subject: "Pass Application",
      html: `<h2>${message}</h2><p>Hi,</p><p>This message is from Pass Application.<wbr>The patient tried to contact you and sent this email.</p><p>Email sent at ${moment().format(
        "MMMM Do YYYY, h:mm:ss a"
      )}</p>`,
    };

    await transporter.sendMail(mailoptions);
    res.status(200).json({
      status: "email sent",
    });
  } catch (err) {
    res.send("failed");
  }
});

app.listen(8000, "127.0.0.1", () => console.log("working"));
